"""Shared-cache multi-arm research: specialists, margin preservation, input-domain denoising.
Frozen native codebook; independent calibration and untouched final speakers.
"""
import argparse,hashlib,json,tempfile,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.signal import stft,istft
from tools.s1_token_aware_restoration import logits
from tools.s1_semantic_robustness import token_metrics
from tools.acoustic_controlled_degradation import degrade

KINDS=("clean","noise10","band8","room")
ARMS=("baseline","spectral-noise","specialist","specialist-margin","routed","routed-margin")
def denoise(x,sr):
    # Conservative spectral subtraction, noise floor estimated from quiet frames.
    _,_,z=stft(x,fs=sr,nperseg=960,noverlap=720)
    power=np.abs(z)**2
    quiet=np.argsort(power.mean(0))[:max(1,power.shape[1]//5)]
    floor=np.median(power[:,quiet],axis=1,keepdims=True)
    gain=np.sqrt(np.maximum(.15,1-1.2*floor/np.maximum(power,1e-12)))
    _,y=istft(z*gain,fs=sr,nperseg=960,noverlap=720)
    y=y[:len(x)]
    if len(y)<len(x):y=np.pad(y,(0,len(x)-len(y)))
    return np.clip(y,-.99,.99).astype(np.float32)
class Expert(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(d,128,1),nn.SiLU(),nn.Conv1d(128,128,7,padding=3),nn.SiLU(),nn.Conv1d(128,d,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        scale=x.detach().square().mean(1,keepdim=True).add(1e-6).sqrt()
        return x+.15*scale*torch.tanh(self.net(x))
def features(x):
    return torch.cat((x.mean(-1),x.std(-1,unbiased=False)),1)
def margin_loss(pred,raw,teacher,scale):
    # Only require a correct winner, not artificially concentrated probabilities.
    t=teacher.reshape(1,-1,1)
    target=pred.gather(-1,t).squeeze(-1)
    other=pred.masked_fill(torch.zeros_like(pred,dtype=torch.bool).scatter_(-1,t,True),-torch.inf).amax(-1)
    # Preserve clean decision boundaries with a modest normalized margin.
    return F.relu((other-target)/scale.squeeze(-1)+.05).mean()
def fit(cache,emb,device,seed):
    torch.manual_seed(seed)
    experts={};hist={}
    for ki,kind in enumerate(("noise10","band8")):
        m=Expert(emb.shape[1]).to(device);opt=torch.optim.AdamW(m.parameters(),lr=2e-4)
        rng=np.random.default_rng(seed+ki);losses=[]
        for step in range(480):
            row=cache[int(rng.integers(len(cache)))]
            c,y=row["clean"];n,_=row[kind]
            c,y,n=c.to(device),y.to(device),n.to(device)
            inp=c if step%3==0 else n
            z=m(inp);raw=logits(inp,emb);p=logits(z,emb)
            scale=raw.detach().std(-1,keepdim=True).clamp_min(1e-6)
            loss=margin_loss(p,raw,y,scale)+.3*(z-c).square().mean()/c.square().mean().clamp_min(1e-6)+.3*(z-inp).square().mean()/inp.square().mean().clamp_min(1e-6)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            assert torch.isfinite(loss);losses.append(float(loss.detach()))
        experts[kind]=m.eval();hist[kind]=dict(updates=480,first30=float(np.mean(losses[:30])),last30=float(np.mean(losses[-30:])))
    gate=nn.Sequential(nn.Linear(emb.shape[1]*2,128),nn.SiLU(),nn.Linear(128,3)).to(device)
    opt=torch.optim.AdamW(gate.parameters(),lr=3e-4);rng=np.random.default_rng(seed)
    for step in range(300):
        row=cache[int(rng.integers(len(cache)))]
        z=torch.cat([features(row[k][0].to(device)) for k in KINDS[:3]],0)
        loss=F.cross_entropy(gate(z),torch.arange(3,device=device))
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return experts,gate.eval(),hist
def selftest():
    x=torch.randn(1,8,20);emb=torch.randn(16,8);m=Expert(8)
    assert torch.equal(m(x),x)
    raw=logits(x,emb);y=raw.argmax(-1).flatten();scale=raw.std(-1,keepdim=True).clamp_min(1e-6)
    assert torch.isfinite(margin_loss(raw,raw,y,scale))
    v=np.random.default_rng(1).normal(0,.05,48000).astype(np.float32);z=denoise(v,48000)
    assert z.shape==v.shape and np.isfinite(z).all()
    print("Specialist, margin and spectral frontend tests passed")
def main(a):
    import soundfile as sf
    from TTS_infer_pack.TTS import TTS,TTS_Config,load_audio_gpu
    from tools.prepare_acoustic_zeroth_pairs import normalize
    manifests=[json.loads((p/"manifest.json").read_text()) for p in (a.train,a.validation,a.test)]
    def items(m):return [i for p in m["pairs"] for i in [p["reference"],*p["targets"]]]
    for i,m in enumerate(manifests):
        assert m["sample_rate"]==48000 and m["revision"]==manifests[0]["revision"]
        for j in range(i):
            assert not {p["speaker_id"] for p in m["pairs"]}&{p["speaker_id"] for p in manifests[j]["pairs"]}
            assert not {normalize(r["text"]) for r in items(m)}&{normalize(r["text"]) for r in items(manifests[j])}
            assert not {r["sha256"] for r in items(m)}&{r["sha256"] for r in items(manifests[j])}
    a.output.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision("highest")
    start=time.monotonic()
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td);cfg=TTS_Config({"custom":dict(TTS_Config.default_configs["v5dev"],device="cuda:0",is_half=False)})
        cfg.configs_path=str(tmp/"tts.yaml");e=TTS(cfg);assert e.precision==torch.float32
        v=e.vits_model.eval().requires_grad_(False);ssl=e.cnhuhbert_model.model.eval().requires_grad_(False);device=e.configs.device
        assert len(v.quantizer.vq.layers)==1
        layer=v.quantizer.vq.layers[0];emb=layer._codebook.embed.detach().float();assert emb.shape[0]==1024
        @torch.no_grad()
        def extract(path):
            wav,_=load_audio_gpu(str(path),16000,device,mono=True,dtype=torch.float32,match_librosa=True)
            h=ssl(wav.reshape(1,-1))["last_hidden_state"].transpose(1,2)
            z=layer.project_in(v.ssl_proj(h).transpose(1,2)).transpose(1,2).contiguous()
            ids=v.extract_latent(h)[0,0].long()
            assert torch.equal(logits(z,emb).argmax(-1).reshape(-1),ids)
            return z.cpu().clone(),ids.cpu().clone()
        def cache(folder,manifest,seed,kinds,with_denoise):
            result=[]
            for j,row in enumerate(items(manifest)):
                path=folder/row["path"];assert hashlib.sha256(path.read_bytes()).hexdigest()==row["pcm_sha256"]
                x,sr=sf.read(path,dtype="float32")
                d={"clean":extract(path)}
                for k,kind in enumerate(kinds):
                    y=degrade(x,sr,kind,seed+j*13+k);dest=tmp/(kind+".wav");sf.write(dest,y,sr,subtype="FLOAT")
                    d[kind]=extract(dest)
                    if with_denoise:
                        sf.write(dest,denoise(y,sr),sr,subtype="FLOAT");d[kind+"-denoise"]=extract(dest)
                if with_denoise:
                    dest=tmp/"clean-denoise.wav";sf.write(dest,denoise(x,sr),sr,subtype="FLOAT");d["clean-denoise"]=extract(dest)
                result.append(d)
            return result
        tr=cache(a.train,manifests[0],10000,("noise10","band8"),False)
        va=cache(a.validation,manifests[1],40000,KINDS[1:],True)
        te=cache(a.test,manifests[2],80000,KINDS[1:],True)
        cache_sec=time.monotonic()-start
        experts,gate,history=fit(tr,emb,device,a.seed)
        fit_sec=time.monotonic()-start-cache_sec
        for k,m in experts.items():torch.save(m.state_dict(),a.output/(k+".pth"))
        torch.save(gate.state_dict(),a.output/"router.pth")
        @torch.no_grad()
        def infer(z):
            z=z.to(device);raw=logits(z,emb);p=gate(features(z)).softmax(-1)[0]
            cls=int(p[1:].argmax())+1;kind=KINDS[cls]
            adapted=logits(experts[kind](z),emb)
            # Only replace low-margin native frames if restored decision is confident.
            old=raw.topk(2,-1);new=adapted.topk(2,-1)
            scale=raw.std(-1).clamp_min(1e-6)
            safe=((old.values[...,0]-old.values[...,1])/scale<.12)&((new.values[...,0]-new.values[...,1])/scale>.05)
            return float(p[cls]),kind,old.indices[...,0].flatten().cpu().numpy(),new.indices[...,0].flatten().cpu().numpy(),torch.where(safe,new.indices[...,0],old.indices[...,0]).flatten().cpu().numpy()
        # Calibrate conservative per-expert thresholds on validation only, protect clean
        # and nonmatching corruption. If a specialist never validates, disable it.
        thresholds={}
        for k in KINDS[1:3]:
            candidates=[]
            for row in va:
                ref=row["clean"][1].numpy()
                for condition in KINDS:
                    p,kind,raw,new,masked=infer(row[condition][0])
                    if kind==k:candidates.append((p,condition,token_metrics(ref,raw)["unit_errors"],token_metrics(ref,masked)["unit_errors"]))
            for threshold in sorted({.5,.7,.8,.9,.95,.99,1.0,*[min(1.,p+1e-6) for p,c,_,_ in candidates if c!=k]}):
                accepted=[q for q in candidates if q[0]>threshold]
                if accepted and sum(q[3]-q[2] for q in accepted)<0 and all(q[3]<=q[2] for q in accepted if q[1]!=k):
                    thresholds[k]=threshold;break
            else:thresholds[k]=1.
        rows=[]
        for j,row in enumerate(te):
            ref=row["clean"][1].numpy()
            for k in KINDS:
                p,kind,raw,new,masked=infer(row[k][0]);active=p>thresholds[kind]
                seqs=(raw,row[k+"-denoise"][1].numpy(),new,masked,new if active else raw,masked if active else raw)
                for arm,seq in zip(ARMS,seqs):
                    rows.append(dict(index=j,kind=k,arm=arm,route=kind,active=active,score=p,**token_metrics(ref,seq)))
        summary={}
        for k in KINDS:
            for arm in ARMS:
                rr=[r for r in rows if r["kind"]==k and r["arm"]==arm]
                summary[k+"-"+arm]=dict(ued=sum(r["unit_errors"] for r in rr)/sum(r["reference_tokens"] for r in rr),
                    activation_rate=float(np.mean([r["active"] for r in rr])),mean_perplexity=float(np.mean([r["candidate"]["perplexity"] for r in rr])))
        report=dict(status="completed",seed=a.seed,thresholds=thresholds,training=history,summary=summary,rows=rows,
                    stage_seconds={"feature_cache":cache_sec,"fit":fit_sec,"total":time.monotonic()-start},
                    caveat="Shared cache; synthetic degradations, no TTS or perceptual proof. Spectral-noise arm is unconditional baseline, not deployment.")
        # Persist restoration metrics before the independent native S1 audit.
        (a.output/"metrics.json").write_text(json.dumps(report,allow_nan=False,indent=2))
        from tools.evaluate_s1_next_waveforms import features as text_features
        from tools.s1_waveform_frontend import ensure_worker_nltk
        ensure_worker_nltk()
        s1_start=time.monotonic();s1_rows=[]
        teacher=e.t2s_model.model.eval()
        with torch.no_grad():
            for i,pair in enumerate(manifests[2]["pairs"]):
                ref=pair["reference"];e.set_ref_audio(str(a.test/ref["path"]))
                prompt=e.prompt_cache["prompt_semantic"].reshape(1,-1).long()
                px,pb=text_features(e,ref["text"],"en")
                for j,target in enumerate(pair["targets"]):
                    tx,tb=text_features(e,target["text"],"en")
                    x=torch.cat((px,tx),1);b=torch.cat((pb,tb),2)
                    expected=te[i*3+j+1]["clean"][1].numpy()
                    for arm,k,penalty in (("native",15,1.35),("topk5",5,1.35),("penalty110",15,1.10)):
                        torch.manual_seed(a.seed+i*1009+j)
                        y,idx=next(teacher.infer_panel_naive(x,torch.tensor([x.size(1)],device=device),prompt,b,
                            top_k=k,top_p=1.,early_stop_num=511,temperature=1.,repetition_penalty=penalty))
                        assert y.ndim==2 and torch.equal(y[:,:prompt.size(1)],prompt)
                        ids=y[:,prompt.size(1):].reshape(-1).cpu().numpy()
                        assert 0<len(ids)<=512
                        s1_rows.append(dict(speaker=i+1,target=j+1,arm=arm,cap_hit=len(ids)>=512,
                            native_stop_idx=int(idx),text_phones=int(tx.size(1)),prompt_tokens=int(prompt.size(1)),
                            **token_metrics(expected,ids)))
        s1_summary={}
        for arm in ("native","topk5","penalty110"):
            rr=[r for r in s1_rows if r["arm"]==arm]
            s1_summary[arm]=dict(cases=len(rr),cap_hits=sum(r["cap_hit"] for r in rr),
                median_length_ratio=float(np.median([r["length_ratio"] for r in rr])),
                mean_ued=sum(r["unit_errors"] for r in rr)/sum(r["reference_tokens"] for r in rr))
        report["s1_audit"]={"rows":s1_rows,"summary":s1_summary,"seconds":time.monotonic()-s1_start,
            "caveat":"Token alignment with natural speech is diagnostic, not generated pronunciation. No synthesized audio."}
        report["stage_seconds"]["total"]=time.monotonic()-start
        (a.output/"metrics.json").write_text(json.dumps(report,allow_nan=False,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true")
    for k in ("train","validation","test","output"):p.add_argument("--"+k,type=Path)
    p.add_argument("--seed",type=int,default=20261127);a=p.parse_args()
    if a.selftest:selftest()
    else:main(a)
