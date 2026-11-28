"""Batched acoustic guidance, time-grid and reference corruption ablations."""
import argparse,gc,hashlib,json,tempfile,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from tools.s1_s2_acoustic_isolation import level_match
from tools.s1_s2_acoustic_cfg import mel_stats

@torch.inference_mode()
def main(a):
    from TTS_infer_pack.TTS import TTS,TTS_Config,mel_fn_v4,norm_spec,denorm_spec,resample,load_audio_gpu
    from tools.s1_waveform_frontend import ensure_worker_nltk
    from tools.evaluate_s1_next_waveforms import features
    from tools.s1_s2_cfm_midpoint import research_v5
    from tools.acoustic_controlled_degradation import degrade
    from module.models_v5 import synthesize_v5_mel
    from tools.acoustic_noise_probe import probe
    from tools.acoustic_periodicity_probe import probe as periodicity
    m=json.loads((a.source/"manifest.json").read_text())
    assert m["verified_same_speaker"] and len(m["pairs"])==4
    lang={"ko":"all_ko","en":"en"}[m.get("language","ko")]
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        def engine_for(version):
            cfg=TTS_Config({"custom":dict(TTS_Config.default_configs[version],device="cuda:0",is_half=False)})
            cfg.configs_path=str(tmp/(version+".yaml"))
            e=TTS(cfg)
            assert e.precision==torch.float32 and e.configs.version==version
            return e
        e=engine_for("v5dev");ensure_worker_nltk()
        v=e.vits_model.eval();device=e.configs.device
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision("highest")
        rate=int(e.vocoder_configs["sr"]);assert rate==48000
        def save(arm,ti,wave,sr,mel=None,**extra):
            if torch.is_tensor(wave):wave=wave.detach().float().cpu().numpy()
            wave=np.asarray(wave,dtype=np.float32).reshape(-1)
            assert len(wave)>=sr/2 and np.isfinite(wave).all() and np.sqrt(np.mean(wave**2))>1e-5
            name=f"ref{ri:02}-text{ti:02}-{arm}"
            sf.write(a.output/(name+".wav"),np.clip(wave,-1,1),sr,subtype="PCM_16")
            leveled,lg=level_match(wave,max_gain=32.)
            sf.write(a.output/(name+"-level.wav"),leveled,sr,subtype="PCM_16")
            row=dict(arm=arm,reference_index=ri,text_index=ti,status="ok",native_rate=sr,
                     noise_proxies=probe(leveled,sr),periodicity=periodicity(leveled,sr),listening_gain=lg,**extra)
            if sr==48000:
                from tools.acoustic_native_bandwidth import audit
                row["wideband"]=audit(wave,sr)
            if mel is not None:
                arr=mel.detach().float().cpu().numpy()
                row.update(mel_stats(arr));np.savez_compressed(a.output/(name+"-mel.npz"),mel=arr)
            rows.append(row)
            (a.output/"metrics.json").write_text(json.dumps(rows,allow_nan=False,indent=2))
            print(json.dumps(row),flush=True)
        original=v.cfm.inference
        for ri in range(a.worker+1,5,2):
            pair=m["pairs"][ri-1]
            ref=pair["reference"];rp=a.source/ref["path"]
            assert all(str(t["speaker_id"])==pair["speaker_id"] for t in [ref,*pair["targets"]])
            raw,rs=sf.read(rp,dtype="float32")
            room=tmp/"room.wav";sf.write(room,degrade(raw,rs,"room",60000+ri),rs,subtype="FLOAT")
            px,pb=features(e,ref["text"],lang)
            e.set_ref_audio(str(rp))
            prompt=e.prompt_cache["prompt_semantic"].reshape(1,-1).long().clone()
            for local,t in enumerate(pair["targets"],1):
                ti=(ri-1)*2+local;target=a.source/t["path"]
                tx,tb=features(e,t["text"],lang);seed=2035+(ri-1)*1009+local
                wav16,_=load_audio_gpu(str(target),16000,device,mono=True,dtype=torch.float32,match_librosa=True)
                ssl=e.cnhuhbert_model.model(wav16.reshape(1,-1))["last_hidden_state"].transpose(1,2)
                oracle=v.extract_latent(ssl)[0,0].reshape(1,1,-1).long()
                token_hash=hashlib.sha256(oracle.cpu().numpy().tobytes()).hexdigest()
                def cond(path):
                    e.set_ref_audio(str(path));entry=e.prompt_cache["refer_spec"][0]
                    spec=(entry[0] if isinstance(entry,tuple) else entry).float().to(device)
                    fr,ge=v.decode_encp(prompt[:,None,:],px,spec)
                    ft,_=v.decode_encp(oracle,tx,spec,ge,1.)
                    r=e.prompt_cache["raw_audio"].float().to(device)
                    if r.shape[0]==2:r=r.mean(0,keepdim=True)
                    if e.prompt_cache["raw_sr"]!=32000:r=resample(r,e.prompt_cache["raw_sr"],32000,device)
                    return fr,ft,norm_spec(mel_fn_v4(r))
                clean=cond(rp);noisy=cond(room)
                def mel_for(c,cfg,interval,sway,noise):
                    torch.manual_seed(seed+100000)
                    calls=[]
                    def solve(*args,**kwargs):
                        d={};y=research_v5(v.cfm,*args,**kwargs,method="euler",guidance_interval=interval,sway=sway,noise_override=noise,diagnostics=d)
                        calls.append(d);return y
                    try:
                        v.cfm.inference=solve
                        mel=denorm_spec(synthesize_v5_mel(v,*c,32,cfg))
                        return mel,{k:sum(d[k] for d in calls) for k in ("conditional_evals","negative_evals")}
                    finally:v.cfm.inference=original
                def vocode(mel):return e.vocoder(mel.float())[0,0]
                real,rr=sf.read(target,dtype="float32");save("real",ti,real,rr)
                wav32=torch.as_tensor(real,device=device)[None]
                if rr!=32000:wav32=resample(wav32,rr,32000,device)
                true=mel_fn_v4(wav32);save("true-mel-vocoder",ti,vocode(true),rate,true)
                for arm,c,cfg,interval,sway,noise in (
                    ("cfg130",clean,1.30,(0.,1.),0.,None),
                    ("cfg090",clean,.90,(0.,1.),0.,None),
                    ("cfg065",clean,.65,(0.,1.),0.,None),
                    ("cfg-mid",clean,1.30,(.25,.75),0.,None),
                    ("cfg-early",clean,1.30,(0.,.5),0.,None),
                    ("cfg-late",clean,1.30,(.5,1.),0.,None),
                    ("sway090",clean,.90,(0.,1.),-1.,None),
                    ("noise100",clean,.90,(0.,1.),0.,1.),
                    ("room090",noisy,.90,(0.,1.),0.,None),
                    ("room-mid",noisy,1.30,(.25,.75),0.,None)):
                    mel,calls=mel_for(c,cfg,interval,sway,noise)
                    save(arm,ti,vocode(mel),rate,mel,token_sha256=token_hash,tokens=int(oracle.size(-1)),
                         cfg=cfg,interval=interval,sway=sway,noise_override=noise,**calls)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--worker",type=int,choices=(0,1),required=True)
    main(p.parse_args())
