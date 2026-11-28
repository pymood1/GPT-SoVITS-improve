"""V24 Korean fixed-token factorial, FP32, no training."""
import argparse,hashlib,json,tempfile,time
from pathlib import Path
import numpy as np
import torch
# cfg, reference target RMS (None=raw), method, steps, guidance interval, noise
ARMS={
 "cfg13":(1.3,None,"euler",32,(0.,1.),.875),
 "cfg145":(1.45,None,"euler",32,(0.,1.),.875),
 "cfg16":(1.6,None,"euler",32,(0.,1.),.875),
 "gain13":(1.3,-23.,"euler",32,(0.,1.),.875),
 "gain145":(1.45,-23.,"euler",32,(0.,1.),.875),
 "gain16":(1.6,-23.,"euler",32,(0.,1.),.875),
 "gain16mid":(1.6,-23.,"midpoint",16,(0.,1.),.875),
 "gain16late":(1.6,-23.,"euler",32,(.25,1.),.875),
 "gain16mild":(1.6,-26.,"euler",32,(0.,1.),.875),
 "gain16noise":(1.6,-23.,"euler",32,(0.,1.),.75),
}
def selftest():
    assert len(ARMS)==10 and ARMS["gain16"][0]==ARMS["cfg16"][0]
    assert {ARMS[a][0] for a in ("cfg13","cfg145","cfg16")}=={ARMS[a][0] for a in ("gain13","gain145","gain16")}
    assert ARMS["gain16mid"][3]*2==ARMS["gain16"][3]
    from tools.s1_s2_cfm_midpoint import selftest as cfm_test
    cfm_test()
    print("V24 factorial and matched NFE controls passed")
def main(a):
    import soundfile as sf
    from TTS_infer_pack.TTS import TTS,TTS_Config
    from tools.s1_waveform_frontend import ensure_worker_nltk
    from tools.s1_s2_acoustic_isolation import level_match
    from tools.acoustic_periodicity_probe import probe as periodicity
    from tools.acoustic_noise_probe import probe as noise
    from tools.s1_s2_cfm_midpoint import research_v5
    a.output.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    start=time.monotonic();rows=[]
    with tempfile.TemporaryDirectory() as td, torch.no_grad():
        tmp=Path(td)
        cfg=TTS_Config({"custom":dict(TTS_Config.default_configs["v5dev"],device="cuda:0",is_half=False)})
        cfg.configs_path=str(tmp/"tts.yaml");e=TTS(cfg);ensure_worker_nltk()
        assert e.precision==torch.float32
        model=e.t2s_model.model.eval();device=e.configs.device
        assert hashlib.file_digest(open(e.configs.t2s_weights_path,"rb"),"sha256").hexdigest()=="87133414860ea14ff6620c483a3db5ed07b44be42e2c3fcdad65523a729a745a"
        state=torch.load(a.weights/"eos4.pt",map_location="cpu",weights_only=True)
        named=dict(model.named_parameters())
        assert sum(named[n].numel() for n in state)==26118656
        for n,p in state.items():
            assert p.dtype==torch.float32 and p.shape==named[n].shape
            named[n].copy_(p)
        e.text_preprocessor.multilingual_bert=None
        manifest=json.loads((a.ko_test/"manifest.json").read_text())
        def frontend(r):
            p=a.ko_test/r["path"]
            assert hashlib.sha256(p.read_bytes()).hexdigest()==r["pcm_sha256"]
            text=r["text"]
            if text[-1] not in ".!?。！？":text+="."
            phones,b,_=e.text_preprocessor.segment_and_extract_feature_for_text(text,"all_ko",e.configs.version)
            x=torch.as_tensor(phones,device=device,dtype=torch.long)[None]
            b=torch.as_tensor(b,device=device,dtype=torch.float32)[None]
            assert b.shape==(1,1024,x.size(1)) and not torch.count_nonzero(b)
            return p,x,b
        for ri,pair in enumerate(manifest["pairs"],1):
            ref,rx,rb=frontend(pair["reference"])
            raw,rs=sf.read(ref,dtype="float32")
            rms_db=20*np.log10(max(1e-12,np.sqrt(np.mean(raw.astype(np.float64)**2))))
            paths={None:ref}
            for db in (-23.,-26.):
                w,_=level_match(raw,target_db=db,max_gain=8.)
                path=tmp/f"ref{ri}-{db}.wav";sf.write(path,w,rs,subtype="FLOAT");paths[db]=path
            e.set_ref_audio(str(ref));prompt=e.prompt_cache["prompt_semantic"].reshape(1,-1).long().clone()
            for ti,target in enumerate(pair["targets"],1):
                real,tx,tb=frontend(target);real_wave,real_sr=sf.read(real,dtype="float32")
                x=torch.cat((rx,tx),1);b=torch.cat((rb,tb),2)
                seed=a.seed+ri*1009+ti;torch.manual_seed(seed)
                y,idx=next(model.infer_panel_naive(x,torch.tensor([x.size(1)],device=device),prompt,b,
                    top_k=15,top_p=1.,early_stop_num=511,temperature=1.,repetition_penalty=1.35))
                assert torch.equal(y[:,:prompt.size(1)],prompt)
                z=y[:,prompt.size(1):].reshape(1,1,-1).long()
                assert 0<z.size(-1)<=512 and z.max()<1024
                tok=hashlib.sha256(z.cpu().numpy().tobytes()).hexdigest()
                # Randomize order deterministically to reduce timing/order confounding.
                order=np.random.default_rng(seed).permutation(list(ARMS))
                for arm in order:
                    arm=str(arm);rate,db,method,steps,interval,scale=ARMS[arm]
                    e.set_ref_audio(str(paths[db]))
                    e.prompt_cache["prompt_semantic"]=prompt.flatten().clone()
                    e.prompt_cache["phones"]=rx[0].tolist()
                    cfm=e.vits_model.cfm;original=cfm.inference;counts=[]
                    def inference(*args,**kw):
                        diag={}
                        out=research_v5(cfm,*args,**kw,method=method,guidance_interval=interval,noise_override=scale,diagnostics=diag)
                        counts.append(diag);return out
                    try:
                        cfm.inference=inference
                        torch.manual_seed(seed+100000)
                        torch.cuda.synchronize();t=time.monotonic()
                        wave=e.using_vocoder_synthesis(z,tx,sample_steps=steps,cfg_rate=rate)
                        torch.cuda.synchronize();elapsed=time.monotonic()-t
                    finally:cfm.inference=original
                    w=wave.detach().float().cpu().numpy().reshape(-1);sr=int(e.vocoder_configs["sr"])
                    assert sr==48000 and len(w)>=sr*.1 and np.isfinite(w).all() and counts
                    level,gain=level_match(w,max_gain=32.)
                    stem=f"ko-ref{ri:02}-target{ti:02}-{arm}"
                    sf.write(a.output/(stem+"-raw.wav"),w,sr,subtype="FLOAT")
                    sf.write(a.output/(stem+"-level.wav"),level,sr,subtype="PCM_16")
                    row=dict(arm=arm,language="ko",reference_index=ri,target_index=ti,token_sha256=tok,cap_hit=z.size(-1)>=512,
                        duration_ratio=(len(w)/sr)/(len(real_wave)/real_sr),noise=noise(w,sr),periodicity=periodicity(w,sr),
                        level_gain=gain,reference_rms_db=float(rms_db),generation_seconds=elapsed,
                        conditional_evals=sum(d["conditional_evals"] for d in counts),negative_evals=sum(d["negative_evals"] for d in counts))
                    rows.append(row)
                    print(json.dumps(dict(event="worker_progress",arm=arm,status="synthesized",language="ko",
                        reference_index=ri,text_index=ti,completed=len(rows),total=len(manifest["pairs"])*2*len(ARMS))),flush=True)
    assert len(rows)==len(manifest["pairs"])*2*len(ARMS)
    (a.output/"metrics.json").write_text(json.dumps(dict(status="completed",seed=a.seed,rows=rows,seconds=time.monotonic()-start),allow_nan=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true")
    for k in ("ko-test","weights","output"):p.add_argument("--"+k,type=Path)
    p.add_argument("--seed",type=int,default=20261129);a=p.parse_args()
    if a.selftest:selftest()
    else:main(a)
