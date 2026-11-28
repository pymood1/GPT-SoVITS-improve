"""Noise-robust SSL residual adapter for existing GPT-SoVITS token vocabulary.
Frozen encoder and quantizer; paired clean/degraded features; speaker holdout.
No replacement of codebook or S1 checkpoint.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tools.acoustic_controlled_degradation import degrade

def token_metrics(reference,candidate,bins=1024):
    a=np.asarray(reference,dtype=np.int64).reshape(-1)
    b=np.asarray(candidate,dtype=np.int64).reshape(-1)
    if not len(a) or not len(b) or min(a.min(),b.min())<0 or max(a.max(),b.max())>=bins:
        raise ValueError("Invalid token sequence")
    prev=list(range(len(b)+1))
    for i,u in enumerate(a,1):
        curr=[i]
        for j,v in enumerate(b,1):
            curr.append(min(prev[j]+1,curr[j-1]+1,prev[j-1]+int(u!=v)))
        prev=curr
    def stats(x):
        p=np.bincount(x,minlength=bins)/len(x)
        p=p[p>0]
        longest=run=0;last=None
        for q in x:
            run=run+1 if q==last else 1
            longest=max(longest,run);last=q
        return dict(unique=int(len(p)),perplexity=float(np.exp(-(p*np.log(p)).sum())),
                    longest_repeat=int(longest))
    return dict(unit_errors=prev[-1],reference_tokens=len(a),
                unit_error_rate=prev[-1]/len(a),length_ratio=len(b)/len(a),
                frame_agreement=float(np.mean(a==b)) if len(a)==len(b) else None,
                reference=stats(a),candidate=stats(b))

class SSLAdapter(nn.Module):
    def __init__(self,channels=768,width=128):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(channels,width,1),nn.SiLU(),
            nn.Conv1d(width,width,5,padding=2),nn.SiLU(),nn.Conv1d(width,channels,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        # Relative bound, per frame. No per-utterance normalization of content.
        scale=x.detach().square().mean(1,keepdim=True).add(1e-6).sqrt()
        return x+.35*scale*torch.tanh(self.net(x))

def fit(examples,device,updates=240,seed=20261127):
    torch.manual_seed(seed);model=SSLAdapter().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    rng=np.random.default_rng(seed);losses=[]
    for step in range(updates):
        clean,noisy=examples[int(rng.integers(len(examples)))]
        clean,noisy=clean.to(device),noisy.to(device)
        if step%4==0:noisy=clean
        pred=model(noisy)
        scale=clean.square().mean().clamp_min(1e-6)
        loss=(pred-clean).square().mean()/scale+.1*(pred-noisy).square().mean()/scale
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite adapter loss")
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        losses.append(float(loss.detach()))
    return model.eval(),dict(updates=updates,first10=float(np.mean(losses[:10])),last10=float(np.mean(losses[-10:])))

def selftest():
    a=token_metrics([0,1,2],[0,1,2]);assert a["unit_error_rate"]==0 and a["frame_agreement"]==1
    assert token_metrics([0,1,2],[0,2])["unit_errors"]==1
    assert token_metrics([0,1,2],[0,9,2])["unit_error_rate"]==1/3
    m=SSLAdapter();x=torch.randn(1,768,12)
    assert torch.equal(m(x),x)
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    for _ in range(3):
        loss=(m(x)-x-.1).square().mean();opt.zero_grad();loss.backward();opt.step()
    d=(m(x)-x).abs()
    assert d.max()>0 and torch.all(d<=.35*x.square().mean(1,keepdim=True).add(1e-6).sqrt()+1e-6)
    print("Token metrics and bounded identity SSL adapter selftest passed")

def main(args):
    import hashlib,tempfile,soundfile as sf
    from TTS_infer_pack.TTS import TTS,TTS_Config,load_audio_gpu
    from tools.prepare_acoustic_zeroth_pairs import normalize
    train=json.loads((args.train/"manifest.json").read_text())
    test=json.loads((args.test/"manifest.json").read_text())
    assert train["sample_rate"]==test["sample_rate"]==48000 and train["revision"]==test["revision"]
    def items(m):
        return [i for p in m["pairs"] for i in [p["reference"],*p["targets"]]]
    assert not {p["speaker_id"] for p in train["pairs"]}&{p["speaker_id"] for p in test["pairs"]}
    assert not {normalize(i["text"]) for i in items(train)}&{normalize(i["text"]) for i in items(test)}
    assert not {i["sha256"] for i in items(train)}&{i["sha256"] for i in items(test)}
    args.output.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td)
        cfg=TTS_Config({"custom":dict(TTS_Config.default_configs["v5dev"],device="cuda:0",is_half=False)})
        cfg.configs_path=str(tmp/"tts.yaml");e=TTS(cfg)
        assert e.precision==torch.float32
        v=e.vits_model.eval().requires_grad_(False);ssl=e.cnhuhbert_model.model.eval().requires_grad_(False);device=e.configs.device
        @torch.no_grad()
        def extract(path,kind,seed):
            if kind!="clean":
                x,sr=sf.read(path,dtype="float32")
                dest=tmp/(path.stem+"-"+kind+"-"+str(seed)+".wav")
                sf.write(dest,degrade(x,sr,kind,seed),sr,subtype="FLOAT")
                path=dest
            wav,_=load_audio_gpu(str(path),16000,device,mono=True,dtype=torch.float32,match_librosa=True)
            h=ssl(wav.reshape(1,-1))["last_hidden_state"].transpose(1,2)
            assert h.shape[1]==768 and torch.isfinite(h).all()
            return h
        @torch.no_grad()
        def tokens(h):
            return v.extract_latent(h)[0,0].long().cpu().numpy()
        pairs=[]
        for j,i in enumerate(items(train)):
            path=args.train/i["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest()==i["pcm_sha256"]
            clean=extract(path,"clean",j)
            for k,kind in enumerate(("noise10","band8")):
                noisy=extract(path,kind,10000+j*13+k)
                assert noisy.shape==clean.shape
                pairs.append((clean.cpu().clone(),noisy.cpu().clone()))
        model,history=fit(pairs,device,seed=args.seed)
        torch.save(model.state_dict(),args.output/"ssl-adapter.pth")
        rows=[]
        for j,i in enumerate(items(test)):
            path=args.test/i["path"];assert hashlib.sha256(path.read_bytes()).hexdigest()==i["pcm_sha256"]
            clean=extract(path,"clean",j);ref=tokens(clean)
            for k,kind in enumerate(("clean","noise10","band8","room")):
                h=extract(path,kind,80000+j*13+k)
                for arm in ("baseline","adapter"):
                    with torch.no_grad():p=h if arm=="baseline" else model(h)
                    row=dict(index=j,kind=kind,arm=arm,**token_metrics(ref,tokens(p)))
                    rows.append(row)
                    print(json.dumps(row,allow_nan=False),flush=True)
        summary={}
        for kind in ("clean","noise10","band8","room"):
            for arm in ("baseline","adapter"):
                rr=[r for r in rows if r["kind"]==kind and r["arm"]==arm]
                summary[kind+"-"+arm]=dict(ued=sum(r["unit_errors"] for r in rr)/sum(r["reference_tokens"] for r in rr),
                    mean_perplexity=float(np.mean([r["candidate"]["perplexity"] for r in rr])),
                    mean_longest_repeat=float(np.mean([r["candidate"]["longest_repeat"] for r in rr])))
        report=dict(status="completed",seed=args.seed,training=history,train_speakers=len(train["pairs"]),test_speakers=len(test["pairs"]),
            train_pairs=len(pairs),test_utterances=len(items(test)),rows=rows,summary=summary,
            caveat="Clean-token agreement is not ground truth phonetic accuracy or perceptual quality. Preserve original codebook. Synthetic corruption and small held-out set.")
        (args.output/"metrics.json").write_text(json.dumps(report,allow_nan=False,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true")
    for k in ("train","test","output"):p.add_argument("--"+k,type=Path)
    p.add_argument("--seed",type=int,default=20261127)
    a=p.parse_args()
    if a.selftest:selftest()
    else:main(a)
