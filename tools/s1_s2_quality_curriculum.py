"""Small frozen-backbone V5 flow-velocity adaptation experiment.
Not full pretraining: train residual only, compare equal updates and held-out speakers.
"""
import argparse,hashlib,json,tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch import nn
from tools.acoustic_controlled_degradation import degrade
from tools.s1_s2_cfm_midpoint import research_v5
from tools.s1_s2_acoustic_isolation import level_match

ARMS=("baseline","clean-only","low-then-clean","robust-then-clean")
CFG=.9
class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(713,128,1),nn.SiLU(),nn.Conv1d(128,128,3,padding=1),
            nn.SiLU(),nn.Conv1d(128,100,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,v,mu,t):
        a=torch.cat((x,v,mu,t[:,None,None].expand(-1,1,x.size(-1))),1)
        return .25*torch.tanh(self.net(a))

def fit(cache,arm,device,seed=20261127):
    torch.manual_seed(seed)
    model=Adapter().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    rng=np.random.default_rng(seed)
    losses=[]
    for step in range(120):
        pool="clean" if arm=="clean-only" or step>=60 else ("low" if arm=="low-then-clean" else "robust")
        # Robust post-stage uses 1/4 low-input/clean-target replay to retain denoising.
        if arm=="robust-then-clean" and step>=60 and step%4==0:pool="robust"
        ex=cache[pool][int(rng.integers(len(cache[pool])))]
        x,v,mu,t,desired,p=[q.to(device) if torch.is_tensor(q) else q for q in ex]
        delta=model(x,v,mu,t)[...,p:]
        loss=(v[...,p:]+delta-desired).square().mean()+.1*delta.square().mean()
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        assert torch.isfinite(loss)
        losses.append(float(loss.detach()))
    return model.eval(),{"updates":len(losses),"first10":float(np.mean(losses[:10])),
                          "last10":float(np.mean(losses[-10:]))}

def selftest():
    torch.manual_seed(1)
    m=Adapter();x=torch.randn(1,100,16);v=torch.randn_like(x);mu=torch.randn(1,512,16);t=torch.tensor([.5])
    assert torch.count_nonzero(m(x,v,mu,t))==0
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    for _ in range(5):
        loss=(m(x,v,mu,t)-.1).square().mean();opt.zero_grad();loss.backward();opt.step()
    y=m(x,v,mu,t)
    assert y.abs().max()<=.25 and y.mean()>0
    print("Bounded zero-init velocity adapter selftest passed")

def main(a):
    from TTS_infer_pack.TTS import TTS,TTS_Config,mel_fn_v4,norm_spec,denorm_spec,resample,load_audio_gpu
    from tools.s1_waveform_frontend import ensure_worker_nltk
    from tools.evaluate_s1_next_waveforms import features
    from module.models_v5 import synthesize_v5_mel,V5_REFERENCE_FRAMES,V5_TOTAL_CHUNK_FRAMES,V5_MAX_TARGET_CHUNK_FRAMES
    from tools.acoustic_native_bandwidth import audit
    from tools.acoustic_noise_probe import probe
    from tools.acoustic_periodicity_probe import probe as periodicity
    train=json.loads((a.train/"manifest.json").read_text());test=json.loads((a.test/"manifest.json").read_text())
    assert train["sample_rate"]==test["sample_rate"]==48000 and train["revision"]==test["revision"]
    assert not {p["speaker_id"] for p in train["pairs"]}&{p["speaker_id"] for p in test["pairs"]}
    a.output.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        cfg=TTS_Config({"custom":dict(TTS_Config.default_configs["v5dev"],device="cuda:0",is_half=False)})
        cfg.configs_path=str(tmp/"tts.yaml")
        e=TTS(cfg);ensure_worker_nltk();v=e.vits_model.eval().requires_grad_(False)
        cfm=v.cfm;device=e.configs.device
        assert e.precision==torch.float32 and e.configs.version=="v5dev"
        def variant(path,kind,seed):
            if kind=="clean":return path
            wave,sr=sf.read(path,dtype="float32")
            dest=tmp/(path.stem+"-"+kind+"-"+str(seed)+".wav")
            sf.write(dest,degrade(wave,sr,kind,seed),sr,subtype="FLOAT")
            return dest
        @torch.no_grad()
        def real_mel(path):
            wave,sr=sf.read(path,dtype="float32");z=torch.as_tensor(wave,device=device)[None]
            if sr!=32000:z=resample(z,sr,32000,device)
            return norm_spec(mel_fn_v4(z))
        @torch.no_grad()
        def condition(rp,tp,ref,target):
            px,_=features(e,ref["text"],"en");tx,_=features(e,target["text"],"en")
            e.set_ref_audio(str(rp))
            prompt=e.prompt_cache["prompt_semantic"].reshape(1,1,-1).long()
            entry=e.prompt_cache["refer_spec"][0]
            spec=(entry[0] if isinstance(entry,tuple) else entry).float().to(device)
            wav16,_=load_audio_gpu(str(tp),16000,device,mono=True,dtype=torch.float32,match_librosa=True)
            ssl=e.cnhuhbert_model.model(wav16.reshape(1,-1))["last_hidden_state"].transpose(1,2)
            ids=v.extract_latent(ssl)[0,0].reshape(1,1,-1).long()
            fr,ge=v.decode_encp(prompt,px,spec);ft,_=v.decode_encp(ids,tx,spec,ge,1.)
            raw=e.prompt_cache["raw_audio"].float().to(device)
            if raw.shape[0]==2:raw=raw.mean(0,keepdim=True)
            if e.prompt_cache["raw_sr"]!=32000:raw=resample(raw,e.prompt_cache["raw_sr"],32000,device)
            return (fr,ft,norm_spec(mel_fn_v4(raw))),ids
        @torch.no_grad()
        def example(c,mel,timepoint,seed):
            fr,ft,rm=c
            p=min(fr.shape[-1],rm.shape[-1]);fr=fr[...,:p][...,-V5_REFERENCE_FRAMES:];rm=rm[...,:p][...,-V5_REFERENCE_FRAMES:]
            p=rm.shape[-1];n=min(ft.shape[-1],mel.shape[-1],V5_TOTAL_CHUNK_FRAMES-p,V5_MAX_TARGET_CHUNK_FRAMES)
            assert n>0
            mu=torch.cat((fr,ft[...,:n]),-1)
            t=torch.tensor([timepoint],device=device)
            noise=torch.randn((1,100,n),device=device,generator=torch.Generator(device=device).manual_seed(seed))
            target=mel[...,:n]
            x=torch.cat((torch.zeros_like(rm),(1-t[:,None,None])*noise+t[:,None,None]*target),-1)
            prompt=torch.cat((rm,torch.zeros_like(target)),-1)
            lengths=torch.tensor([p+n],device=device)
            args=dict(text0=mu,use_grad_ckpt=False,drop_text=False,infer=True)
            vel,emb,_=cfm.estimator(x,prompt,lengths,t,drop_audio_cond=False,**args)
            neg,_,_=cfm.estimator(x,prompt,lengths,t,drop_audio_cond=True,text_cache=emb,**args)
            vel=vel.transpose(2,1);vel=vel+CFG*(vel-neg.transpose(2,1))
            return tuple(z.detach().cpu().clone() if torch.is_tensor(z) else z for z in (x,vel,mu,t,target-noise,p))
        cache={k:[] for k in ("clean","low","robust")}
        for si,pair in enumerate(train["pairs"]):
            ref=pair["reference"];rp=a.train/ref["path"]
            for j,target in enumerate(pair["targets"]):
                tp=a.train/target["path"];seed=10000+si*101+j
                clean=real_mel(tp);c,_=condition(rp,tp,ref,target)
                for q,t in enumerate((.2,.5,.8)):cache["clean"].append(example(c,clean,t,seed+q))
                for k,kind in enumerate(("noise10","band8")):
                    r=variant(rp,kind,seed+k);f=variant(tp,kind,seed+k+7)
                    low=real_mel(f);lc,_=condition(r,f,ref,target)
                    for q,t in enumerate((.2,.5,.8)):
                        cache["low"].append(example(lc,low,t,seed+q))
                        cache["robust"].append(example(lc,clean,t,seed+q))
                print(json.dumps({"event":"train_cache","speaker":si+1,"target":j+1}),flush=True)
        # no inference-mode tensors in adapter autograd
        cache={k:[tuple(z.clone() if torch.is_tensor(z) else z for z in ex) for ex in pool] for k,pool in cache.items()}
        models={"baseline":None};history={}
        for arm in ARMS[1:]:
            models[arm],history[arm]=fit(cache,arm,device)
            torch.save(models[arm].state_dict(),a.output/(arm+".pth"))
        (a.output/"training.json").write_text(json.dumps({"cache_counts":{k:len(v) for k,v in cache.items()},"arms":history},indent=2))
        rows=[];pair=test["pairs"][a.worker];ri=a.worker+1
        ref=pair["reference"];rp=a.test/ref["path"]
        for j,target in enumerate(pair["targets"],1):
            ti=(ri-1)*2+j;tp=a.test/target["path"];seed=2035+a.worker*1009+j
            real,sr=sf.read(tp,dtype="float32")
            for kind in ("clean","room"):
                r=variant(rp,kind,seed);f=variant(tp,kind,seed+7)
                c,ids=condition(r,f,ref,target)
                tok=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
                for arm in ARMS:
                    name=f"{arm}-{kind}"
                    original=cfm.inference
                    try:
                        cfm.inference=lambda *args,**kw:research_v5(cfm,*args,**kw,method="euler",velocity_adapter=models[arm])
                        torch.manual_seed(seed+100000)
                        with torch.no_grad():
                            mel=denorm_spec(synthesize_v5_mel(v,*c,32,CFG));wave=e.vocoder(mel.float())[0,0].float().cpu().numpy()
                    finally:cfm.inference=original
                    leveled,_=level_match(wave,max_gain=32.)
                    base=f"ref{ri:02}-text{ti:02}-{name}"
                    sf.write(a.output/(base+".wav"),np.clip(wave,-1,1),48000,subtype="PCM_16")
                    sf.write(a.output/(base+"-level.wav"),leveled,48000,subtype="PCM_16")
                    row=dict(arm=name,reference_index=ri,text_index=ti,status="ok",native_rate=48000,
                             token_sha256=tok,tokens=int(ids.size(-1)),noise_proxies=probe(leveled,48000),
                             periodicity=periodicity(leveled,48000),wideband=audit(wave,48000))
                    rows.append(row)
                    (a.output/"metrics.json").write_text(json.dumps(rows,allow_nan=False,indent=2))
                    print(json.dumps(row,allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    for k in ("train","test","output"):p.add_argument("--"+k,type=Path)
    p.add_argument("--worker",type=int,choices=(0,1),default=0)
    a=p.parse_args()
    if a.selftest:selftest()
    else:main(a)
