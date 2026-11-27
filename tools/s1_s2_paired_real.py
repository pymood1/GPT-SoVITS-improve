"""Paired real-target fixed-token S2 factorial, plus oracle and V2 native controls."""
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
    from tools.train_s1_rms_polish import Generator
    from module.models_v5 import synthesize_v5_mel
    from tools.acoustic_noise_probe import probe
    from tools.acoustic_periodicity_probe import probe as periodicity
    m=json.loads((a.source/"manifest.json").read_text())
    assert m["verified_same_speaker"] and len(m["pairs"])==2
    pair=m["pairs"][a.worker];ri=a.worker+1
    ref=pair["reference"];rp=a.source/ref["path"]
    assert all(str(t["speaker_id"])==pair["speaker_id"] for t in [ref,*pair["targets"]])
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
        teacher=e.t2s_model.model.eval();v=e.vits_model.eval();device=e.configs.device
        rate=int(e.vocoder_configs["sr"]);assert rate==48000
        raw,rs=sf.read(rp,dtype="float32");normalized,gain=level_match(raw)
        gp=tmp/"gain.wav";sf.write(gp,normalized,rs,subtype="FLOAT")
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
            if mel is not None:
                arr=mel.detach().float().cpu().numpy()
                row.update(mel_stats(arr));np.savez_compressed(a.output/(name+"-mel.npz"),mel=arr)
            rows.append(row)
            (a.output/"metrics.json").write_text(json.dumps(rows,allow_nan=False,indent=2))
            print(json.dumps(row),flush=True)
        for local,t in enumerate(pair["targets"],1):
            ti=(ri-1)*2+local
            target=a.source/t["path"]
            px,pb=features(e,ref["text"]);tx,tb=features(e,t["text"])
            e.set_ref_audio(str(rp));e.prompt_cache["phones"]=px[0].tolist()
            prompt=e.prompt_cache["prompt_semantic"].reshape(1,-1).long().clone()
            x=torch.cat((px,tx),1);b=torch.cat((pb,tb),2)
            seed=2035+(ri-1)*1009+local
            ids,_=Generator(teacher,{}).generate(x,b,prompt,max_new=512,min_new=10,top_k=15,
                         repetition_penalty=1.35,generator=torch.Generator(device=device).manual_seed(seed),cached=False)
            assert ids.numel()>0 and int(ids.max())<1024
            ids=ids[:,None,:];token_hash=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
            def cond(path,tokens):
                e.set_ref_audio(str(path))
                entry=e.prompt_cache["refer_spec"][0]
                spec=(entry[0] if isinstance(entry,tuple) else entry).float().to(device)
                # Freeze raw reference prompt tokens, not just target S1 tokens.
                fr,ge=v.decode_encp(prompt[:,None,:],px,spec)
                ft,_=v.decode_encp(tokens,tx,spec,ge,1.)
                r=e.prompt_cache["raw_audio"].float().to(device)
                if r.shape[0]==2:r=r.mean(0,keepdim=True)
                if e.prompt_cache["raw_sr"]!=32000:r=resample(r,e.prompt_cache["raw_sr"],32000,device)
                return fr,ft,norm_spec(mel_fn_v4(r))
            def mel_for(c):
                torch.manual_seed(seed+100000)
                return denorm_spec(synthesize_v5_mel(v,*c,32,1.30))
            def vocode(mel):return e.vocoder(mel.float())[0,0]
            base=mel_for(cond(rp,ids));boost=mel_for(cond(gp,ids))
            for arm,mel in (("raw",base),("raw-hard",base.clamp(-12,2)),("gain",boost),("gain-hard",boost.clamp(-12,2))):
                save(arm,ti,vocode(mel),rate,mel,token_sha256=token_hash,tokens=int(ids.size(-1)),cap_hit=ids.size(-1)==512)
            real,rr=sf.read(target,dtype="float32")
            save("real",ti,real,rr)
            wav32=torch.as_tensor(real,device=device)[None]
            if rr!=32000:wav32=resample(wav32,rr,32000,device)
            true=mel_fn_v4(wav32)
            save("true-mel-vocoder",ti,vocode(true),rate,true)
            wav16,_=load_audio_gpu(str(target),16000,device,mono=True,dtype=torch.float32,match_librosa=True)
            ssl=e.cnhuhbert_model.model(wav16.reshape(1,-1))["last_hidden_state"].transpose(1,2)
            oracle=v.extract_latent(ssl)[0,0].reshape(1,1,-1).long()
            om=mel_for(cond(rp,oracle))
            save("oracle-s2",ti,vocode(om),rate,om,tokens=int(oracle.size(-1)))
        del e,v,teacher;gc.collect();torch.cuda.empty_cache()
        e=engine_for("v2ProPlus")
        for local,t in enumerate(pair["targets"],1):
            ti=(ri-1)*2+local;seed=2035+(ri-1)*1009+local
            result=list(e.run(dict(text=t["text"],text_lang="all_ko",ref_audio_path=str(rp),
                prompt_text=ref["text"],prompt_lang="all_ko",top_k=15,top_p=1.,temperature=1.,
                repetition_penalty=1.35,seed=seed,batch_size=1,parallel_infer=False,split_bucket=False,
                use_cuda_graph=False,use_flash_attention=False,text_split_method="cut0",fragment_interval=.1,
                return_fragment=False,streaming_mode=False)))
            assert len(result)==1
            sr,w=result[0];assert np.issubdtype(w.dtype,np.integer)
            save("v2ProPlus",ti,w.astype(np.float32)/32768,sr)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--worker",type=int,choices=(0,1),required=True)
    main(p.parse_args())
