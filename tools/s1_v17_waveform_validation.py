"""Frozen v17 checkpoints, paired bilingual waveform evaluation. No retraining."""
import argparse,hashlib,json,tempfile,time
from pathlib import Path
import numpy as np
import torch
from tools.s1_semantic_robustness import token_metrics
ARMS=("real","oracle","frozen","eos2","eos4","eos4_ctc")
def cases(m,lang):
    # Fixed first/last target of each English speaker; all Korean targets.
    return [(ri,ti) for ri,p in enumerate(m["pairs"],1)
            for ti in (range(1,len(p["targets"])+1) if lang=="ko" else (1,len(p["targets"])))]
def selftest():
    m={"pairs":[{"targets":[0,1,2,3]}]*4}
    assert len(cases(m,"en"))==8 and cases(m,"en")[:2]==[(1,1),(1,4)]
    assert len(cases({"pairs":[{"targets":[0,1]}]*2},"ko"))==4
    print("Fixed paired bilingual evaluation selection selftest passed")
@torch.inference_mode()
def main(a):
    import soundfile as sf
    from TTS_infer_pack.TTS import TTS,TTS_Config,load_audio_gpu
    from tools.evaluate_s1_next_waveforms import features
    from tools.s1_waveform_frontend import ensure_worker_nltk
    from tools.s1_s2_acoustic_isolation import level_match
    from tools.acoustic_noise_probe import probe as noise
    from tools.acoustic_periodicity_probe import probe as periodicity
    a.output.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    start=time.monotonic()
    with tempfile.TemporaryDirectory() as td:
        cfg=TTS_Config({"custom":dict(TTS_Config.default_configs["v5dev"],device="cuda:0",is_half=False)})
        cfg.configs_path=str(Path(td)/"tts.yaml");e=TTS(cfg);ensure_worker_nltk()
        assert e.precision==torch.float32 and e.configs.version=="v5dev"
        assert hashlib.file_digest(open(e.configs.t2s_weights_path,"rb"),"sha256").hexdigest()=="87133414860ea14ff6620c483a3db5ed07b44be42e2c3fcdad65523a729a745a"
        model=e.t2s_model.model.eval();device=e.configs.device
        named=dict(model.named_parameters())
        expected={n for n,p in named.items() if n.startswith(("ar_text_embedding.","ar_predict_layer.")) or any(n.startswith(f"h.layers.{i}.") for i in range(len(model.h.layers)-8,len(model.h.layers)))}
        assert sum(named[n].numel() for n in expected)==26118656
        initial={n:named[n].detach().cpu().clone() for n in expected}
        data=[]
        for folder,lang in ((a.en,"en"),(a.ko,"ko")):
            m=json.loads((folder/"manifest.json").read_text())
            assert m["sample_rate"]==(48000 if lang=="en" else 16000)
            def check(r):
                p=folder/r["path"];assert hashlib.sha256(p.read_bytes()).hexdigest()==r["pcm_sha256"];return p
            for ri,ti in cases(m,lang):
                pair=m["pairs"][ri-1];ref=pair["reference"];t=pair["targets"][ti-1]
                rp=check(ref);tp=check(t);frontend="en" if lang=="en" else "all_ko"
                e.set_ref_audio(str(rp))
                prompt=e.prompt_cache["prompt_semantic"].reshape(1,-1).long().clone()
                px,pb=features(e,ref["text"],frontend);tx,tb=features(e,t["text"],frontend)
                w,_=load_audio_gpu(str(tp),16000,device,mono=True,dtype=torch.float32,match_librosa=True)
                h=e.cnhuhbert_model.model(w.reshape(1,-1))["last_hidden_state"].transpose(1,2)
                ids=e.vits_model.extract_latent(h)[0,0].reshape(1,-1).long()
                real,sr=sf.read(tp,dtype="float32")
                assert sr==m["sample_rate"] and ids.max()<1024
                data.append(dict(lang=lang,ri=ri,ti=ti,rp=rp,tp=tp,px=px,tx=tx,prompt=prompt,ids=ids,
                    x=torch.cat((px,tx),1),b=torch.cat((pb,tb),2),real=real,sr=sr))
        assert len(data)==12
        rows=[];checks={}
        for arm in ARMS:
            if arm in ("real","oracle","frozen"):state=initial
            else:
                path=a.weights/(arm+".pt");state=torch.load(path,map_location="cpu",weights_only=True)
                assert set(state)==expected and all(state[n].shape==named[n].shape and state[n].dtype==torch.float32 and torch.isfinite(state[n]).all() for n in state)
                checks[arm]=hashlib.file_digest(open(path,"rb"),"sha256").hexdigest()
            for n,p in state.items():named[n].copy_(p)
            for r in data:
                lang,ri,ti=r["lang"],r["ri"],r["ti"]
                seed=a.seed+ri*1009+ti
                row=dict(arm=arm,language=lang,reference_index=ri,target_index=ti,seed=a.seed)
                if arm=="real":
                    wave,sr=r["real"],r["sr"]
                else:
                    if arm=="oracle":z=r["ids"].flatten().cpu().numpy()
                    else:
                        torch.manual_seed(seed)
                        y,idx=next(model.infer_panel_naive(r["x"],torch.tensor([r["x"].size(1)],device=device),r["prompt"],r["b"],
                            top_k=15,top_p=1.,early_stop_num=511,temperature=1.,repetition_penalty=1.35))
                        assert torch.equal(y[:,:r["prompt"].size(1)],r["prompt"])
                        z=y[:,r["prompt"].size(1):].flatten().cpu().numpy()
                        assert 0<len(z)<=512 and z.max()<1024
                        row["cap_hit"]=len(z)>=512
                    row.update(token_metrics(r["ids"].flatten().cpu().numpy(),z))
                    row["generated_tokens"]=len(z)
                    e.set_ref_audio(str(r["rp"]));e.prompt_cache["phones"]=r["px"][0].tolist()
                    assert torch.equal(e.prompt_cache["prompt_semantic"].reshape(1,-1).long(),r["prompt"])
                    torch.manual_seed(seed+100000)
                    wav=e.using_vocoder_synthesis(torch.as_tensor(z,device=device,dtype=torch.long).reshape(1,1,-1),r["tx"],sample_steps=32,cfg_rate=.90)
                    wave=wav.detach().float().cpu().numpy().reshape(-1);sr=int(e.vocoder_configs["sr"])
                    assert sr==48000 and len(wave)>=sr*.1
                assert np.isfinite(wave).all()
                stem=f"{lang}-ref{ri:02}-target{ti:02}-{arm}"
                row["duration_ratio"]=(len(wave)/sr)/(len(r["real"])/r["sr"])
                row["raw_peak"]=float(np.max(np.abs(wave)))
                row["noise"]=noise(wave,sr);row["periodicity"]=periodicity(wave,sr)
                level,gain=level_match(wave,max_gain=32.);row["level_gain"]=gain
                # Float raw retains original amplitude without clipping on write.
                sf.write(a.output/(stem+"-raw.wav"),wave,sr,subtype="FLOAT")
                sf.write(a.output/(stem+"-level.wav"),level,sr,subtype="PCM_16")
                rows.append(row)
                print(json.dumps(dict(arm=arm,status="synthesized",language=lang,reference_index=ri,text_index=ti,
                    completed=len(rows),total=len(ARMS)*len(data),duration_ratio=row["duration_ratio"])),flush=True)
            (a.output/"metrics.json").write_text(json.dumps(dict(status="partial",rows=rows,weight_hashes=checks),allow_nan=False))
        assert len(rows)==72
        (a.output/"metrics.json").write_text(json.dumps(dict(status="completed",seed=a.seed,rows=rows,weight_hashes=checks,seconds=time.monotonic()-start),allow_nan=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true")
    for k in ("en","ko","weights","output"):p.add_argument("--"+k,type=Path)
    p.add_argument("--seed",type=int,default=20261127);a=p.parse_args()
    if a.selftest:selftest()
    else:main(a)
