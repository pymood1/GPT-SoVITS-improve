"""v19: matched S1 text-contrastive continued training and fixed-token S2 guidance.
Experimental only; no production changes."""
import argparse,hashlib,json,tempfile,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tools.s1_expanded_alignment_pilot import pack,loss_fn
from tools.s1_eos_root_cause import summary_logits
from tools.s1_semantic_robustness import token_metrics
ARMS_EN=("base","continued","contrastive")
ARMS_KO=("base","gain","late_cfg","gain_late")
def rank_loss(pos,neg,margin=.25):
    return F.softplus(neg-pos+margin).mean()
def logprob(lg,labels,starts,lengths):
    p=torch.arange(labels.size(1),device=lg.device)[None]
    m=(p>=starts[:,None])&(p<lengths[:,None])
    v=lg.log_softmax(-1).gather(-1,labels.unsqueeze(-1)).squeeze(-1)
    return (v*m).sum(1)/m.sum(1).clamp_min(1)
def selftest():
    x=torch.tensor([-2.,-3.],requires_grad=True);y=torch.tensor([-3.,-2.],requires_grad=True)
    rank_loss(x,y).backward();assert (x.grad<0).all() and (y.grad>0).all()
    logits=torch.zeros(1,5,3,requires_grad=True);labels=torch.zeros(1,5,dtype=torch.long)
    logprob(logits,labels,torch.tensor([1]),torch.tensor([3])).sum().backward()
    assert logits.grad[0,[0,3,4]].abs().sum()==0 and logits.grad[0,1:3].abs().sum()>0
    print("Text contrastive sign and true-target masking selftests passed")
def main(a):
    import soundfile as sf
    from TTS_infer_pack.TTS import TTS,TTS_Config,load_audio_gpu
    from tools.evaluate_s1_next_waveforms import features
    from tools.s1_waveform_frontend import ensure_worker_nltk
    from tools.s1_s2_acoustic_isolation import level_match
    from tools.acoustic_periodicity_probe import probe as periodicity
    from tools.acoustic_noise_probe import probe as noise
    from tools.s1_s2_cfm_midpoint import research_v5
    a.output.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    start=time.monotonic()
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td)
        cfg=TTS_Config({"custom":dict(TTS_Config.default_configs["v5dev"],device="cuda:0",is_half=False)})
        cfg.configs_path=str(tmp/"tts.yaml");e=TTS(cfg);ensure_worker_nltk()
        assert e.precision==torch.float32
        model=e.t2s_model.model.eval();device=e.configs.device
        assert hashlib.file_digest(open(e.configs.t2s_weights_path,"rb"),"sha256").hexdigest()=="87133414860ea14ff6620c483a3db5ed07b44be42e2c3fcdad65523a729a745a"
        for p in model.parameters():p.requires_grad_(False)
        for mod in (*model.h.layers[-8:],model.ar_text_embedding,model.ar_predict_layer):
            for p in mod.parameters():p.requires_grad_(True)
        named={n:p for n,p in model.named_parameters() if p.requires_grad}
        assert sum(p.numel() for p in named.values())==26118656
        def load(arm):
            state=torch.load(a.weights/(arm+".pt"),map_location="cpu",weights_only=True)
            assert set(state)==set(named) and all(state[n].dtype==torch.float32 and state[n].shape==p.shape for n,p in named.items())
            with torch.no_grad():
                for n,p in named.items():p.copy_(state[n])
        @torch.no_grad()
        def cache(folder,lang):
            m=json.loads((folder/"manifest.json").read_text());pool=[];pairs=[]
            frontend="en" if lang=="en" else "all_ko"
            def one(r):
                p=folder/r["path"];assert hashlib.sha256(p.read_bytes()).hexdigest()==r["pcm_sha256"]
                x,b=features(e,r["text"],frontend)
                w,_=load_audio_gpu(str(p),16000,device,mono=True,dtype=torch.float32,match_librosa=True)
                h=e.cnhuhbert_model.model(w.reshape(1,-1))["last_hidden_state"].transpose(1,2)
                y=e.vits_model.extract_latent(h)[0,0].reshape(1,-1).long()
                return dict(x=x,b=b,y=y,path=p,lang=lang,start=0,phones=x)
            for ri,p in enumerate(m["pairs"],1):
                ref=one(p["reference"]);pool.append(ref)
                e.set_ref_audio(str(ref["path"]))
                prompt=e.prompt_cache["prompt_semantic"].reshape(1,-1).long().clone()
                for ti,t in enumerate(p["targets"],1):
                    r=one(t);pool.append(r)
                    q=dict(x=torch.cat((ref["x"],r["x"]),1),b=torch.cat((ref["b"],r["b"]),2),
                        y=torch.cat((prompt,r["y"]),1),start=prompt.size(1),phones=r["x"],lang=lang,
                        ref=ref,target=r,prompt=prompt,ri=ri,ti=ti)
                    pairs.append(q)
            return pool,pairs
        ep,et=cache(a.en_train,"en");kp,kt=cache(a.ko_train,"ko")
        _,val=cache(a.en_val,"en");_,test=cache(a.en_test,"en");_,ko=cache(a.ko_test,"ko")
        assert len(ep)==80 and len(kp)==32 and len(val)==16 and len(test)==16 and len(ko)==8
        enpool=ep+et;kopool=kp+kt
        def forward(rows):
            x,xl,y,yl,b=pack(rows,device,True)
            xy,mask,labels=model.make_input_data(x,xl,y,yl,b)
            h,_=model.h((xy,None),mask=mask)
            lg=model.ar_predict_layer(h[:,x.size(1)-1:])
            starts=torch.tensor([r["start"] for r in rows],device=device)
            return lg,labels,starts,yl
        # For prompted positives, keep the exact reference prefix and replace only target phones.
        def wrong(r,pool,rng):
            candidates=[p for p in pool if p["path"]!=(r["target"]["path"] if "target" in r else r["path"])
                        and ("ref" not in r or p["path"]!=r["ref"]["path"])
                        and not torch.equal(p["x"],r["phones"])]
            assert candidates
            distance=min(abs(p["x"].size(1)-r["phones"].size(1)) for p in candidates)
            near=[p for p in candidates if abs(p["x"].size(1)-r["phones"].size(1))==distance][:8]
            p=near[int(rng.integers(len(near)))];q=dict(r)
            q["x"]=torch.cat((r["ref"]["x"],p["x"]),1) if "ref" in r else p["x"]
            q["b"]=torch.cat((r["ref"]["b"],p["b"]),2) if "ref" in r else p["b"]
            return q
        rng=np.random.default_rng(a.seed+4000)
        batches=[]
        for _ in range(a.steps):
            rr=[enpool[i] for i in rng.choice(len(enpool),3,replace=False)]+[kopool[int(rng.integers(len(kopool)))]]
            # English-only contrastive term; Korean remains CE replay.
            batches.append((rr,[wrong(r,ep,rng) for r in rr[:3]]))
        rows=[];diagnostic=[];trace={}
        def save(r,arm,wave,sr,**extra):
            wave=np.asarray(wave,dtype=np.float32).reshape(-1)
            real,rs=sf.read(r["target"]["path"],dtype="float32")
            assert len(wave)>=sr*.1 and np.isfinite(wave).all()
            level,gain=level_match(wave,max_gain=32.)
            name=f"{r['lang']}-ref{r['ri']:02}-target{r['ti']:02}-{arm}"
            sf.write(a.output/(name+"-level.wav"),level,sr,subtype="PCM_16")
            sf.write(a.output/(name+"-raw.wav"),wave,sr,subtype="FLOAT")
            row=dict(arm=arm,language=r["lang"],reference_index=r["ri"],target_index=r["ti"],
                duration_ratio=(len(wave)/sr)/(len(real)/rs),noise=noise(wave,sr),
                periodicity=periodicity(wave,sr),level_gain=gain,**extra)
            rows.append(row)
            print(json.dumps(dict(arm=arm,status="synthesized",language=r["lang"],
                reference_index=r["ri"],text_index=r["ti"],duration_ratio=row["duration_ratio"])),flush=True)
        @torch.no_grad()
        def generate(r):
            seed=a.seed+r["ri"]*1009+r["ti"];torch.manual_seed(seed)
            y,idx=next(model.infer_panel_naive(r["x"],torch.tensor([r["x"].size(1)],device=device),r["prompt"],r["b"],
                top_k=15,top_p=1.,early_stop_num=511,temperature=1.,repetition_penalty=1.35))
            assert torch.equal(y[:,:r["start"]],r["prompt"])
            z=y[:,r["start"]:].reshape(1,1,-1).long()
            assert 0<z.size(-1)<=512 and z.max()<1024
            return z,seed
        @torch.no_grad()
        def synth(r,z,seed,path=None,late=False):
            e.set_ref_audio(str(path or r["ref"]["path"]))
            # Keep semantic conditioning identical even when acoustic reference gain changes.
            e.prompt_cache["prompt_semantic"]=r["prompt"].flatten().clone()
            e.prompt_cache["phones"]=r["ref"]["x"][0].tolist()
            cfm=e.vits_model.cfm;original=cfm.inference
            try:
                if late:cfm.inference=lambda *args,**kw:research_v5(cfm,*args,**kw,method="euler",guidance_interval=(.5,1.))
                torch.manual_seed(seed+100000)
                wav=e.using_vocoder_synthesis(z,r["target"]["x"],sample_steps=32,cfg_rate=1.3 if late else .9)
                return wav.detach().float().cpu().numpy(),int(e.vocoder_configs["sr"])
            finally:cfm.inference=original
        @torch.no_grad()
        def evaluate(arm):
            for split,rr in (("validation",val),("test",test)):
                for r in rr:
                    lg,lbl,starts,yl=forward([r]);p=r["start"]
                    d=summary_logits(lg[:,p:],lbl[:,p:],model.EOS)
                    diagnostic.append(dict(arm=arm,split=split,speaker=r["ri"],target=r["ti"],**d))
                    if split=="test" and r["ti"] in (1,4):
                        z,seed=generate(r);w,sr=synth(r,z,seed);assert sr==48000
                        save(r,arm,w,sr,cap_hit=z.size(-1)>=512,**token_metrics(r["target"]["y"].flatten().cpu().numpy(),z.flatten().cpu().numpy()))
        for arm in ARMS_EN:
            load("eos2");trace[arm]=[]
            if arm!="base":
                opt=torch.optim.AdamW(list(named.values()),lr=1e-5,weight_decay=.01)
                for step,(rr,negative) in enumerate(batches,1):
                    opt.zero_grad(set_to_none=True)
                    lg,lbl,st,yl=forward(rr+negative)
                    # Identical padded geometry across arms and matched/wrong text examples.
                    ce=loss_fn(lg[:4],lbl[:4],yl[:4],True,2.,st[:4])
                    if arm=="contrastive":
                        scores=logprob(lg,lbl,st,yl)
                        pos=scores[:3];neg=scores[4:]
                        ranking=rank_loss(pos,neg);loss=ce+.3*ranking
                    else:ranking=ce.detach()*0;loss=ce
                    assert torch.isfinite(loss)
                    loss.backward();torch.nn.utils.clip_grad_norm_(list(named.values()),1.,error_if_nonfinite=True);opt.step()
                    if step==1 or step%20==0 or step==a.steps:
                        event=dict(arm=arm,status="training",reference_index=step,text_index=a.steps,
                            ce=float(ce.detach()),ranking=float(ranking.detach()))
                        trace[arm].append(event);print(json.dumps(event),flush=True)
                del opt
                torch.save({n:p.detach().cpu() for n,p in named.items()},a.output/(arm+".pt"))
            evaluate(arm)
        # Separate Korean acoustic study: fixed EOS4 weights and fixed generated tokens.
        load("eos4")
        for r in ko:
            z,seed=generate(r);tok=hashlib.sha256(z.cpu().numpy().tobytes()).hexdigest()
            ref,rs=sf.read(r["ref"]["path"],dtype="float32")
            scaled,gain=level_match(ref,target_db=-23.,max_gain=8.)
            path=tmp/f"ko-{r['ri']}.wav";sf.write(path,scaled,rs,subtype="FLOAT")
            for arm in ARMS_KO:
                w,sr=synth(r,z,seed,path if "gain" in arm else None,"late" in arm)
                assert sr==48000
                save(r,arm,w,sr,token_sha256=tok,reference_gain=gain if "gain" in arm else 1.,cap_hit=z.size(-1)>=512)
        assert len(rows)==56 and len(diagnostic)==96
        (a.output/"metrics.json").write_text(json.dumps(dict(status="completed",seed=a.seed,rows=rows,diagnostic=diagnostic,
            training_trace=trace,seconds=time.monotonic()-start),allow_nan=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true")
    for k in ("en-train","en-val","en-test","ko-train","ko-test","weights","output"):p.add_argument("--"+k,type=Path)
    p.add_argument("--seed",type=int,default=20261127);p.add_argument("--steps",type=int,default=160);a=p.parse_args()
    if a.selftest:selftest()
    else:main(a)
