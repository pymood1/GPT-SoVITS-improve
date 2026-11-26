"""Frozen V5dev acoustic isolation: true mel, oracle codes, reference gain,
CFM CFG and static cache controls. No training. Save raw and level-matched WAVs."""
import argparse
import json
import math
import time
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
import torch


def db(x):
    return 20*math.log10(max(float(x),1e-12))


def stats(x,sr):
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Invalid audio")
    n=max(1,int(sr*.02))
    frames=x[:len(x)//n*n].reshape(-1,n)
    rms=np.sqrt(np.mean(x*x))
    quiet=np.sqrt(np.mean(frames*frames,axis=1))
    # A lower-percentile frame statistic, not a speech/noise classifier.
    p10=float(np.quantile(quiet,.1)) if len(quiet) else rms
    return {"seconds":len(x)/sr,"rms_dbfs":db(rms),"peak_dbfs":db(np.max(np.abs(x))),
            "quiet_p10_dbfs":db(p10),"dc":float(x.mean()),
            "clipped_fraction":float(np.mean(np.abs(x)>=1.))}


def level_match(x,target_db=-23.,max_gain=8.):
    x=np.asarray(x,dtype=np.float32)
    peak=float(np.max(np.abs(x)))
    rms=float(np.sqrt(np.mean(x.astype(np.float64)**2)))
    if not np.isfinite(x).all() or rms<1e-8:
        raise ValueError("Invalid normalization input")
    gain=min(10**(target_db/20)/rms,max_gain,.95/max(peak,1e-8))
    return x*gain,float(gain)


def selftest():
    x=np.sin(np.arange(48000)*2*np.pi*440/48000).astype(np.float32)*.01
    y,g=level_match(x)
    assert g<=8 and abs(stats(y,48000)["seconds"]-1)<1e-9
    assert abs(stats(y,48000)["rms_dbfs"]-stats(x,48000)["rms_dbfs"]-db(g))<1e-5
    assert np.max(np.abs(y))<=.95
    print("Acoustic diagnostic level and statistics tests passed")


@torch.inference_mode()
def main(a):
    from TTS_infer_pack.TTS import TTS,TTS_Config,mel_fn_v4,norm_spec,denorm_spec,resample,load_audio_gpu
    from tools.s1_waveform_frontend import ensure_worker_nltk
    from tools.evaluate_s1_next_waveforms import features
    from tools.train_s1_rms_polish import Generator
    from module.models_v5 import synthesize_v5_mel

    m=json.loads(a.manifest.read_text())
    ri=a.worker_index
    # Both references get the same target, enabling cross-reference controls.
    ti=0
    ref=m["references"][ri];target=m["targets"][ti]
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        cfg=TTS_Config({"custom":dict(TTS_Config.default_configs["v5dev"],device="cuda:0",
              is_half=False,t2s_weights_path=str(a.teacher.resolve()),vits_weights_path=str(a.s2.resolve()))})
        cfg.configs_path=str(Path(tmp)/"tts.yaml")
        engine=TTS(cfg)
        ensure_worker_nltk()
        teacher=engine.t2s_model.model.eval().requires_grad_(False)
        vits=engine.vits_model.eval()
        device=next(teacher.parameters()).device
        assert engine.precision==torch.float32 and engine.configs.version=="v5dev"
        assert int(engine.vocoder_configs["sr"])==48000

        def save(name,wave,sr=48000,mel=None,**extra):
            if torch.is_tensor(wave):
                wave=wave.detach().float().cpu().numpy()
            wave=np.asarray(wave,dtype=np.float32).reshape(-1)
            row={"arm":name,"reference_index":ri+1,"text_index":ti+1,"status":"ok",
                 **stats(wave,sr),**extra}
            sf.write(a.output/(name+".wav"),np.clip(wave,-1,1),sr,subtype="PCM_16")
            leveled,gain=level_match(wave,max_gain=32.)
            sf.write(a.output/(name+"-level.wav"),leveled,sr,subtype="PCM_16")
            row["listening_gain"]=gain
            if mel is not None:
                arr=mel.detach().float().cpu().numpy()
                if not np.isfinite(arr).all():
                    raise ValueError("Nonfinite mel")
                np.savez_compressed(a.output/(name+"-mel.npz"),mel=arr)
                row.update(mel_min=float(arr.min()),mel_max=float(arr.max()),mel_mean=float(arr.mean()),
                           mel_frames=int(arr.shape[-1]))
            rows.append(row)
            (a.output/"metrics.json").write_text(json.dumps(rows,allow_nan=False,indent=2))
            print(json.dumps(row,allow_nan=False),flush=True)

        def load(path,sr):
            wav,rate=sf.read(path,dtype="float32")
            if wav.ndim==2:
                wav=wav.mean(axis=1)
            x=torch.as_tensor(wav,device=device)[None]
            if rate!=sr:
                x=resample(x,rate,sr,device)
            return x

        def oracle_mel(path):
            return mel_fn_v4(load(path,32000))

        def vocode(mel):
            return engine.vocoder(mel.to(engine.precision))[0,0]

        for label,item in (("reference",ref),("target",target)):
            raw,sr=sf.read(item["path"],dtype="float32")
            save(label+"-real",raw,sr)
            mel=oracle_mel(item["path"])
            save(label+"-true-mel-vocoder",vocode(mel),mel=mel)

        # Real target semantics (no GPT sampling, no reference padding).
        wav16,_=load_audio_gpu(target["path"],16000,device,mono=True,dtype=torch.float32,match_librosa=True)
        ssl=engine.cnhuhbert_model.model(wav16.reshape(1,-1))["last_hidden_state"].transpose(1,2)
        codes=vits.extract_latent(ssl)[0,0].reshape(1,1,-1).long()
        assert codes.numel()>0 and int(codes.max())<1024
        px,pb=features(engine,ref["text"])
        tx,tb=features(engine,target["text"])
        engine.set_ref_audio(ref["path"])
        engine.prompt_cache["phones"]=px[0].tolist()
        prompt=engine.prompt_cache["prompt_semantic"].reshape(1,-1).long()
        x=torch.cat((px,tx),1);b=torch.cat((pb,tb),2)
        seed=2035+ri*1009+ti
        gen=Generator(teacher,{})
        ids,_=gen.generate(x,b,prompt,max_new=384,min_new=10,top_k=15,repetition_penalty=1.35,
                           generator=torch.Generator(device=device).manual_seed(seed),cached=False)
        assert ids.numel()>0 and int(ids.max())<1024
        ids=ids[:,None,:]
        np.savez_compressed(a.output/"semantic-tokens.npz",oracle=codes.cpu().numpy(),generated=ids.cpu().numpy())
        default_cache=bool(vits.cfm.use_static_cache)
        def acoustic(name,tokens,cfg=1.30,cache=default_cache):
            vits.cfm.use_static_cache=cache
            prompt_ids=engine.prompt_cache["prompt_semantic"].reshape(1,1,-1)
            raw_entry=engine.prompt_cache["refer_spec"][0]
            spec=(raw_entry[0] if isinstance(raw_entry,tuple) else raw_entry).float().to(device)
            fea_ref,ge=vits.decode_encp(prompt_ids,px,spec)
            fea,_=vits.decode_encp(tokens,tx,spec,ge,1.)
            raw=engine.prompt_cache["raw_audio"].float().to(device)
            if raw.shape[0]==2:
                raw=raw.mean(0,keepdim=True)
            sr=engine.prompt_cache["raw_sr"]
            if sr!=32000:
                raw=resample(raw,sr,32000,device)
            refmel=norm_spec(mel_fn_v4(raw))
            torch.manual_seed(seed+100000)
            mel=denorm_spec(synthesize_v5_mel(vits,fea_ref,fea,refmel,32,cfg))
            save(name,vocode(mel),mel=mel,tokens=int(tokens.size(-1)),cfg=cfg,static_cache=cache)
            vits.cfm.use_static_cache=default_cache

        acoustic("oracle-s2",codes)
        acoustic("teacher-baseline",ids)
        acoustic("teacher-cfg0",ids,cfg=0.)
        acoustic("teacher-cache-off",ids,cache=False)
        raw,sr=sf.read(ref["path"],dtype="float32")
        leveled,gain=level_match(raw)
        refpath=str(Path(tmp)/"normalized-reference.wav")
        sf.write(refpath,leveled,sr,subtype="FLOAT")
        engine.set_ref_audio(refpath)
        engine.prompt_cache["phones"]=px[0].tolist()
        acoustic("teacher-ref-gain",ids)
        save("reference-gain-input",leveled,sr,applied_gain=gain)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    for key in ("teacher","s2","manifest","output"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--worker-index",type=int,default=0)
    a=p.parse_args()
    if a.selftest:
        selftest()
    else:
        main(a)
