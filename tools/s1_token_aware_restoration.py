"""Frozen-codebook token-aware restoration and learned utterance quality gate.
Uses clean teacher IDs, never changes base model weights or vocabulary.
"""
import argparse,hashlib,json,tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tools.s1_semantic_robustness import token_metrics
from tools.acoustic_controlled_degradation import degrade

def logits(x,emb):
    # Exact frozen Euclidean codebook geometry; omit common x norm for CE.
    z=x.transpose(1,2)
    return 2*z@emb.T-emb.square().sum(-1)
class Model(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.restore=nn.Sequential(nn.Conv1d(d,256,1),nn.SiLU(),nn.Conv1d(256,256,7,padding=3),nn.SiLU(),nn.Conv1d(256,d,1))
        nn.init.zeros_(self.restore[-1].weight);nn.init.zeros_(self.restore[-1].bias)
        self.gate=nn.Sequential(nn.Linear(2*d,128),nn.SiLU(),nn.Linear(128,1))
    def forward(self,x):
        scale=x.detach().square().mean(1,keepdim=True).add(1e-6).sqrt()
        return x+.5*scale*torch.tanh(self.restore(x))
    def quality(self,x):
        # Learned corruption detector, no test labels or knowledge of corruption type.
        return self.gate(torch.cat((x.mean(-1),x.std(-1,unbiased=False)),1)).flatten()
def fit(pairs,emb,device,seed):
    torch.manual_seed(seed);m=Model(emb.shape[1]).to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.01)
    rng=np.random.default_rng(seed);losses=[]
    for step in range(720):
        c,n,y=[a.to(device) for a in pairs[int(rng.integers(len(pairs)))]]
        inp=c if step%3==0 else n
        z=m(inp)
        l=logits(z,emb)
        scale=logits(inp,emb).detach().std(-1,keepdim=True).clamp_min(1e-6)
        ce=F.cross_entropy((4*l/scale).flatten(0,1),y.reshape(-1))
        rms=c.square().mean().clamp_min(1e-6)
        feat=(z-c).square().mean()/rms
        preserve=(z-inp).square().mean()/rms
        loss=ce+.2*feat+(.5 if step%3==0 else .05)*preserve
        q=torch.cat((m.quality(c),m.quality(n)))
        loss=loss+F.binary_cross_entropy_with_logits(q,torch.tensor([0.,1.],device=device))
        if not torch.isfinite(loss):raise FloatingPointError("nonfinite training")
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        losses.append(float(loss.detach()))
    return m.eval(),dict(updates=len(losses),first30=float(np.mean(losses[:30])),last30=float(np.mean(losses[-30:])))
def selftest():
    torch.manual_seed(11);x=torch.randn(1,8,17);emb=torch.randn(16,8)
    assert torch.equal(logits(x,emb).argmax(-1),torch.cdist(x.transpose(1,2),emb).argmin(-1))
    m=Model(8);assert torch.equal(m(x),x)
    y=logits(x,emb).argmax(-1);loss=F.cross_entropy(logits(m(x+.1),emb).flatten(0,1),y.flatten())
    loss.backward();assert m.restore[-1].weight.grad.abs().sum()>0
    assert m.quality(x).shape==(1,)
    print("Frozen-codebook CE and identity restoration selftest passed")
def main(a):
    import soundfile as sf
    from TTS_infer_pack.TTS import TTS,TTS_Config,load_audio_gpu
    from tools.prepare_acoustic_zeroth_pairs import normalize
    manifests=[json.loads((p/"manifest.json").read_text()) for p in (a.train,a.validation,a.test)]
    def items(m):return [i for p in m["pairs"] for i in [p["reference"],*p["targets"]]]
    assert all(m["sample_rate"]==48000 and m["revision"]==manifests[0]["revision"] for m in manifests)
    for i in range(3):
        for j in range(i):
            assert not {p["speaker_id"] for p in manifests[i]["pairs"]}&{p["speaker_id"] for p in manifests[j]["pairs"]}
            assert not {normalize(r["text"]) for r in items(manifests[i])}&{normalize(r["text"]) for r in items(manifests[j])}
            assert not {r["sha256"] for r in items(manifests[i])}&{r["sha256"] for r in items(manifests[j])}
    a.output.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td);cfg=TTS_Config({"custom":dict(TTS_Config.default_configs["v5dev"],device="cuda:0",is_half=False)})
        cfg.configs_path=str(tmp/"tts.yaml");e=TTS(cfg);assert e.precision==torch.float32
        v=e.vits_model.eval().requires_grad_(False);ssl=e.cnhuhbert_model.model.eval().requires_grad_(False);device=e.configs.device
        assert len(v.quantizer.vq.layers)==1
        layer=v.quantizer.vq.layers[0];emb=layer._codebook.embed.detach().float()
        assert emb.shape[0]==1024 and torch.isfinite(emb).all()
        @torch.no_grad()
        def extract(path,kind,seed):
            if kind!="clean":
                x,sr=sf.read(path,dtype="float32");dest=tmp/(path.stem+"-"+kind+"-"+str(seed)+".wav")
                sf.write(dest,degrade(x,sr,kind,seed),sr,subtype="FLOAT");path=dest
            wav,_=load_audio_gpu(str(path),16000,device,mono=True,dtype=torch.float32,match_librosa=True)
            h=ssl(wav.reshape(1,-1))["last_hidden_state"].transpose(1,2)
            z=layer.project_in(v.ssl_proj(h).transpose(1,2)).transpose(1,2).contiguous()
            ids=v.extract_latent(h)[0,0].long()
            # Verify exact equivalence to original quantizer, fail instead of guessing.
            assert torch.equal(logits(z,emb).argmax(-1).reshape(-1),ids)
            return z,ids
        def checked(folder,row):
            p=folder/row["path"]
            assert hashlib.sha256(p.read_bytes()).hexdigest()==row["pcm_sha256"]
            return p
        pairs=[]
        for j,row in enumerate(items(manifests[0])):
            path=checked(a.train,row);c,y=extract(path,"clean",j)
            for k,kind in enumerate(("noise10","band8")):
                n,_=extract(path,kind,10000+j*13+k);assert n.shape==c.shape
                pairs.append((c.cpu().clone(),n.cpu().clone(),y.cpu().clone()))
        model,history=fit(pairs,emb,device,a.seed)
        torch.save(model.state_dict(),a.output/"token-aware.pth")
        # Calibrate threshold ONLY using prior v10 validation split. No test kind labels.
        val=[]
        with torch.no_grad():
            for j,row in enumerate(items(manifests[1])):
                p=checked(a.validation,row)
                for k,kind in enumerate(("clean","noise10","band8","room")):
                    z,_=extract(p,kind,40000+j*13+k)
                    val.append(dict(index=j,kind=kind,score=float(model.quality(z).sigmoid().item())))
        threshold=min(1.,max(r["score"] for r in val if r["kind"]=="clean")+.02)
        rows=[]
        with torch.no_grad():
            for j,row in enumerate(items(manifests[2])):
                path=checked(a.test,row);c,y=extract(path,"clean",j);ref=y.cpu().numpy()
                for k,kind in enumerate(("clean","noise10","band8","room")):
                    z,ids=extract(path,kind,80000+j*13+k)
                    new=logits(model(z),emb).argmax(-1).reshape(-1).cpu().numpy()
                    score=float(model.quality(z).sigmoid().item());raw=ids.cpu().numpy()
                    for arm,seq in (("baseline",raw),("token-aware",new),("gated",new if score>threshold else raw)):
                        rows.append(dict(index=j,kind=kind,arm=arm,score=score,gate_active=score>threshold,**token_metrics(ref,seq)))
        summary={}
        for kind in ("clean","noise10","band8","room"):
            for arm in ("baseline","token-aware","gated"):
                rr=[r for r in rows if r["kind"]==kind and r["arm"]==arm]
                summary[kind+"-"+arm]=dict(ued=sum(r["unit_errors"] for r in rr)/sum(r["reference_tokens"] for r in rr),
                    activation_rate=float(np.mean([r["gate_active"] for r in rr])),
                    mean_perplexity=float(np.mean([r["candidate"]["perplexity"] for r in rr])),
                    max_repeat=max(r["candidate"]["longest_repeat"] for r in rr))
        report=dict(status="completed",seed=a.seed,training=history,threshold=threshold,validation=val,rows=rows,summary=summary,
                    train_speakers=len(manifests[0]["pairs"]),validation_speakers=len(manifests[1]["pairs"]),test_speakers=len(manifests[2]["pairs"]),
                    test_utterances=len(items(manifests[2])),caveat="Synthetic distortions, clean token teacher not phonetic ground truth. No synthesis.")
        (a.output/"metrics.json").write_text(json.dumps(report,allow_nan=False,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true")
    for k in ("train","validation","test","output"):p.add_argument("--"+k,type=Path)
    p.add_argument("--seed",type=int,default=20261127);a=p.parse_args()
    if a.selftest:selftest()
    else:main(a)
