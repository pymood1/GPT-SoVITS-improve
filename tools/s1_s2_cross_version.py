"""Native-checkpoint cross-version comparison on fixed text/reference pairs.
V5 gain+hard is end-to-end (gain also changes S1 prompt), not S2 isolation.
"""
import argparse, gc, importlib, json, tempfile, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

def main(a):
    from TTS_infer_pack.TTS import TTS,TTS_Config
    from tools.s1_waveform_frontend import ensure_worker_nltk
    from tools.s1_s2_acoustic_isolation import level_match
    from tools.acoustic_noise_probe import probe
    mod=importlib.import_module("TTS_infer_pack.TTS")
    m=json.loads((a.source/"texts.json").read_text())
    assert len(m["references"])==2 and len(m["targets"])>=3
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    versions=("v5dev",) if a.worker==0 else ("v2ProPlus","v5turbo")
    ensure_worker_nltk()
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        for version in versions:
            cfg=TTS_Config({"custom":dict(TTS_Config.default_configs[version],device="cuda:0",is_half=False)})
            cfg.configs_path=str(tmp/(version+".yaml"))
            engine=TTS(cfg)
            assert engine.configs.version==version and engine.precision==torch.float32
            for ri in (1,2):
                ref=a.source/f"reference-{ri:02}.wav"
                assert ref.is_file()
                wave,sr=sf.read(ref,dtype="float32")
                normalized,gain=level_match(wave)
                normalized_path=tmp/f"gain-{ri}.wav"
                sf.write(normalized_path,normalized,sr,subtype="FLOAT")
                arms=("v5dev","v5dev-gain-hard") if version=="v5dev" else (version,)
                for ti in (2,3):
                    for arm in arms:
                        seed=2035+(ri-1)*1009+(ti-1)
                        modified=arm.endswith("gain-hard")
                        old=mod.denorm_spec
                        calls=[0]
                        def clipped(x):
                            calls[0]+=1
                            return old(x).clamp(-12,2)
                        if modified:mod.denorm_spec=clipped
                        started=time.monotonic()
                        try:
                            result=list(engine.run(dict(text=m["targets"][ti-1],text_lang="all_ko",
                                ref_audio_path=str(normalized_path if modified else ref),
                                prompt_text=m["references"][ri-1],prompt_lang="all_ko",
                                top_k=15,top_p=1.,temperature=1.,repetition_penalty=1.35,
                                seed=seed,batch_size=1,parallel_infer=False,split_bucket=False,
                                use_cuda_graph=False,use_flash_attention=False,
                                text_split_method="cut0",fragment_interval=.1,
                                return_fragment=False,streaming_mode=False,
                                sample_steps=4 if version=="v5turbo" else 32,
                                cfg_rate=1.3 if version=="v5dev" else 0.)))
                        finally:mod.denorm_spec=old
                        assert len(result)==1
                        rate,audio=result[0]
                        if modified:assert calls[0]>0,"Clipping path not exercised"
                        x=np.asarray(audio)
                        assert x.ndim==1 and np.issubdtype(x.dtype,np.integer)
                        x=x.astype(np.float32)/32768
                        assert rate>=16000 and len(x)>=rate/2 and np.isfinite(x).all() and np.sqrt(np.mean(x*x))>1e-4
                        name=f"ref{ri:02}-text{ti:02}-{arm}"
                        sf.write(a.output/(name+".wav"),x,rate,subtype="PCM_16")
                        leveled,_=level_match(x,max_gain=32.)
                        sf.write(a.output/(name+"-level.wav"),leveled,rate,subtype="PCM_16")
                        row=dict(arm=arm,reference_index=ri,text_index=ti,status="ok",seed=seed,
                            native_rate=int(rate),gain=float(gain) if modified else 1.,clamp_calls=calls[0],
                            seconds=time.monotonic()-started,noise_proxies=probe(leveled,rate))
                        rows.append(row)
                        (a.output/"metrics.json").write_text(json.dumps(rows,allow_nan=False,indent=2))
                        print(json.dumps(row),flush=True)
            del engine,cfg
            gc.collect();torch.cuda.empty_cache()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--worker",type=int,choices=(0,1),required=True)
    main(p.parse_args())
