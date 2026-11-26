"""Second acoustic pilot: isolate CFG, mel outliers and reference gain pathways.
Reuse exact saved tokens, source audio and baseline mel, no S1 inference/training.
"""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from tools.s1_s2_acoustic_isolation import stats,level_match

def mel_stats(m):
    a=np.asarray(m,dtype=np.float32)
    if not a.size or not np.isfinite(a).all():
        raise ValueError("Invalid mel")
    return {"mel_min":float(a.min()),"mel_max":float(a.max()),
            "mel_p001":float(np.quantile(a,.001)),"mel_p01":float(np.quantile(a,.01)),
            "mel_below_minus12":float(np.mean(a< -12)),
            "mel_above_2":float(np.mean(a>2))}

def selftest():
    m=np.array([-49.,-12.,-11.,0.,3.],np.float32)
    r=mel_stats(m)
    assert abs(r["mel_below_minus12"]-.2)<1e-6 and abs(r["mel_above_2"]-.2)<1e-6
    assert np.clip(m,-12,2).tolist()==[-12.,-12.,-11.,0.,2.]
    print("Acoustic mel range diagnostics passed")

@torch.inference_mode()
def main(a):
    from TTS_infer_pack.TTS import TTS,TTS_Config,mel_fn_v4,norm_spec,denorm_spec,resample
    from tools.s1_waveform_frontend import ensure_worker_nltk
    from tools.evaluate_s1_next_waveforms import features
    from module.models_v5 import synthesize_v5_mel
    ri=a.worker_index
    source=a.source/f"ref{ri+1:02}"
    texts=json.loads(a.texts.read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        cfg=TTS_Config({"custom":dict(TTS_Config.default_configs["v5dev"],device="cuda:0",
            is_half=False,t2s_weights_path=str(a.teacher.resolve()),vits_weights_path=str(a.s2.resolve()))})
        cfg.configs_path=str(Path(tmp)/"tts.yaml")
        engine=TTS(cfg)
        ensure_worker_nltk()
        assert engine.precision==torch.float32 and engine.configs.version=="v5dev"
        device=engine.configs.device
        vits=engine.vits_model.eval()
        sr=engine.vocoder_configs["sr"]
        assert sr==48000
        def save(name,mel,**extra):
            arr=mel.detach().float().cpu().numpy()
            wave=engine.vocoder(mel.float())[0,0].float().cpu().numpy()
            row={"arm":name,"reference_index":ri+1,"text_index":1,"status":"ok",
                 **stats(wave,sr),**mel_stats(arr),**extra}
            sf.write(a.output/(name+".wav"),np.clip(wave,-1,1),sr,subtype="PCM_16")
            leveled,gain=level_match(wave,max_gain=32.)
            sf.write(a.output/(name+"-level.wav"),leveled,sr,subtype="PCM_16")
            row["listening_gain"]=gain
            np.savez_compressed(a.output/(name+"-mel.npz"),mel=arr)
            rows.append(row)
            (a.output/"metrics.json").write_text(json.dumps(rows,indent=2,allow_nan=False))
            print(json.dumps(row,allow_nan=False),flush=True)

        # Saved mel controls isolate vocoder; no new CFM samples or RNG.
        mel=torch.from_numpy(np.load(source/"teacher-baseline-mel.npz")["mel"]).to(device)
        save("saved-baseline",mel)
        save("saved-clamp",mel.clamp(-12,2))
        # Verify that an exact saved mel reproduces previous raw PCM up to quantization.
        prior,_=sf.read(source/"teacher-baseline.wav",dtype="float32")
        now,_=sf.read(a.output/"saved-baseline.wav",dtype="float32")
        assert prior.shape==now.shape
        rows[-2]["saved_pcm_max_delta"]=float(np.max(np.abs(prior-now)))
        (a.output/"metrics.json").write_text(json.dumps(rows,indent=2,allow_nan=False))
        ids=torch.from_numpy(np.load(source/"semantic-tokens.npz")["generated"]).to(device).long()
        assert ids.ndim==3 and ids.shape[:2]==(1,1) and ids.numel()>0 and int(ids.max())<1024
        px,_=features(engine,texts["references"][ri])
        tx,_=features(engine,texts["targets"][0])
        def conditions(path):
            engine.set_ref_audio(str(path))
            prompt=engine.prompt_cache["prompt_semantic"].reshape(1,1,-1)
            entry=engine.prompt_cache["refer_spec"][0]
            spec=(entry[0] if isinstance(entry,tuple) else entry).float().to(device)
            ref,ge=vits.decode_encp(prompt,px,spec)
            target,_=vits.decode_encp(ids,tx,spec,ge,1.)
            raw=engine.prompt_cache["raw_audio"].float().to(device)
            if raw.shape[0]==2:raw=raw.mean(0,keepdim=True)
            rate=engine.prompt_cache["raw_sr"]
            if rate!=32000:raw=resample(raw,rate,32000,device)
            return ref,target,norm_spec(mel_fn_v4(raw))
        base=conditions(source/"reference-real.wav")
        # Use the same 8x cap as the preceding reference gain test.
        raw,rate=sf.read(source/"reference-real.wav",dtype="float32")
        gainwav,gain=level_match(raw)
        path=Path(tmp)/"gain.wav"
        sf.write(path,gainwav,rate,subtype="FLOAT")
        normalized=conditions(path)
        assert base[0].shape==normalized[0].shape and base[1].shape==normalized[1].shape and base[2].shape==normalized[2].shape
        seed=2035+ri*1009+100000
        def synth(name,cond,cfg,**extra):
            torch.manual_seed(seed)
            m=denorm_spec(synthesize_v5_mel(vits,*cond,32,cfg))
            save(name,m,cfg=cfg,**extra)
        synth("raw-cfg050",base,.5)
        synth("raw-cfg080",base,.8)
        synth("gain-cfg080",normalized,.8,applied_gain=gain)
        synth("mel-gain-only-cfg080",(base[0],base[1],normalized[2]),.8,applied_gain=gain)
        synth("encoder-gain-only-cfg080",(normalized[0],normalized[1],base[2]),.8,applied_gain=gain)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    for name in ("teacher","s2","source","texts","output"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--worker-index",type=int,default=0)
    a=p.parse_args()
    if a.selftest:selftest()
    else:main(a)
