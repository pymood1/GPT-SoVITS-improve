"""v20: frozen multilingual BERT conditioning and matched S2 guidance schedules."""
import argparse,hashlib,json,tempfile,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tools.s1_expanded_alignment_pilot import pack,loss_fn
from tools.s1_eos_root_cause import summary_logits
from tools.s1_semantic_robustness import token_metrics
ARMS_EN=("base","zero","bert","bert_reverse")
ARMS_KO=("native09","full09","full13","late09","late13")
def rank_loss(pos,neg,margin=.25):
    return F.softplus(neg-pos+margin).mean()
def logprob(lg,labels,starts,lengths):
    p=torch.arange(labels.size(1),device=lg.device)[None]
    m=(p>=starts[:,None])&(p<lengths[:,None])
    v=lg.log_softmax(-1).gather(-1,labels.unsqueeze(-1)).squeeze(-1)
    return (v*m).sum(1)/m.sum(1).clamp_min(1)
def bert_features(b,mode):
    if mode=="zero":return torch.zeros_like(b)
    if mode=="bert":return b
    if mode=="bert_reverse":return b.flip(-1)
    raise ValueError(mode)
def selftest():
    b=torch.arange(12).reshape(1,3,4).float()
    assert torch.equal(bert_features(b,"bert_reverse"),b.flip(-1)) and not torch.count_nonzero(bert_features(b,"zero"))
    x=torch.tensor([-2.,-3.],requires_grad=True);y=torch.tensor([-3.,-2.],requires_grad=True)
    rank_loss(x,y).backward();assert (x.grad<0).all() and (y.grad>0).all()
    logits=torch.zeros(1,5,3,requires_grad=True);labels=torch.zeros(1,5,dtype=torch.long)
    logprob(logits,labels,torch.tensor([1]),torch.tensor([3])).sum().backward()
    assert logits.grad[0,[0,3,4]].abs().sum()==0 and logits.grad[0,1:3].abs().sum()>0
    orig=torch.nn.Linear(1024,8)
    bridge=torch.nn.Linear(768,8,bias=False)
    torch.nn.init.zeros_(bridge.weight)
    z=torch.zeros(1,4,1024);expected=orig(z).detach()
    h=orig.register_forward_hook(lambda m,args,out:m.bias+bridge(args[0][...,:768]))
    assert torch.equal(orig(z),expected)
    orig(torch.ones_like(z)).sum().backward()
    assert torch.count_nonzero(bridge.weight.grad)>0
    h.remove()
    print("Multilingual bridge identity, gradient and feature ordering selftests passed")
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
        from text.multilingual_bert import MultilingualBertFeatureExtractor
        extractor=MultilingualBertFeatureExtractor(str(a.bert_model),device=device,is_half=False)
        e.text_preprocessor.multilingual_bert=extractor
        assert all(p.dtype==torch.float32 for p in extractor.model.parameters())
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
                assert b.shape==(1,1024,x.size(1)) and torch.count_nonzero(b)>0 and torch.isfinite(b).all()
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
        ep,et=cache(a.en_train,"en");extra,extra_pairs=cache(a.en_extra,"en");kp,kt=cache(a.ko_train,"ko")
        _,val=cache(a.en_val,"en");_,test=cache(a.en_test,"en");_,ko=cache(a.ko_test,"ko")
        assert len(ep)==80 and len(extra)==80 and len(kp)==32 and len(val)==16 and len(test)==16 and len(ko)==8
        e.text_preprocessor.multilingual_bert=None
        del extractor
        torch.cuda.empty_cache()
        enpool=ep+et+extra+extra_pairs;kopool=kp+kt
        mode="zero"
        # Preserve the original zero-feature bias; a zero-init bridge maps frozen 768D mBERT features.
        bridge=torch.nn.Linear(768,model.bert_proj.out_features,bias=False).to(device)
        torch.nn.init.zeros_(bridge.weight)
        def hook(module,inp,out):
            return module.bias+bridge(inp[0][...,:768])
        with torch.no_grad():
            z=torch.zeros(1,2,1024,device=device);expected_zero=model.bert_proj(z).clone()
        handle=model.bert_proj.register_forward_hook(hook)
        with torch.no_grad():torch.testing.assert_close(model.bert_proj(z),expected_zero,atol=0.,rtol=0.)
        def conditioned(r):
            q=dict(r)
            if "ref" in r:
                q["b"]=torch.cat((bert_features(r["ref"]["b"],mode),bert_features(r["target"]["b"],mode)),2)
            else:q["b"]=bert_features(r["b"],mode)
            return q
        def forward(rows):
            x,xl,y,yl,b=pack([conditioned(r) for r in rows],device,True)
            xy,mask,labels=model.make_input_data(x,xl,y,yl,b)
            h,_=model.h((xy,None),mask=mask)
            lg=model.ar_predict_layer(h[:,x.size(1)-1:])
            starts=torch.tensor([r["start"] for r in rows],device=device)
            return lg,labels,starts,yl
        rng=np.random.default_rng(a.seed+4000)
        batches=[]
        for _ in range(a.steps):
            rr=[enpool[i] for i in rng.choice(len(enpool),3,replace=False)]+[kopool[int(rng.integers(len(kopool)))]]
            batches.append(rr)
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
            y,idx=next(model.infer_panel_naive(r["x"],torch.tensor([r["x"].size(1)],device=device),r["prompt"],conditioned(r)["b"],
                top_k=15,top_p=1.,early_stop_num=511,temperature=1.,repetition_penalty=1.35))
            assert torch.equal(y[:,:r["start"]],r["prompt"])
            z=y[:,r["start"]:].reshape(1,1,-1).long()
            assert 0<z.size(-1)<=512 and z.max()<1024
            return z,seed
        @torch.no_grad()
        def synth(r,z,seed,acoustic="native09"):
            e.set_ref_audio(str(r["ref"]["path"]))
            # Keep semantic conditioning identical even when acoustic reference gain changes.
            e.prompt_cache["prompt_semantic"]=r["prompt"].flatten().clone()
            e.prompt_cache["phones"]=r["ref"]["x"][0].tolist()
            cfm=e.vits_model.cfm;original=cfm.inference
            try:
                if acoustic!="native09":
                    interval=(.5,1.) if acoustic.startswith("late") else (0.,1.)
                    cfm.inference=lambda *args,**kw:research_v5(cfm,*args,**kw,method="euler",guidance_interval=interval)
                torch.manual_seed(seed+100000)
                wav=e.using_vocoder_synthesis(z,r["target"]["x"],sample_steps=32,cfg_rate=1.3 if acoustic.endswith("13") else .9)
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
            mode=arm if arm in ("bert","bert_reverse") else "zero"
            with torch.no_grad():bridge.weight.zero_()
            if arm!="base":
                params=list(named.values())+list(bridge.parameters())
                opt=torch.optim.AdamW(params,lr=2e-5,weight_decay=.01)
                for step,rr in enumerate(batches,1):
                    opt.zero_grad(set_to_none=True)
                    lg,lbl,st,yl=forward(rr);ce=loss_fn(lg,lbl,yl,True,2.,st)
                    loss=ce
                    assert torch.isfinite(loss)
                    loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
                    if step==1 or step%20==0 or step==a.steps:
                        event=dict(arm=arm,status="training",reference_index=step,text_index=a.steps,
                            ce=float(ce.detach()),bridge_norm=float(bridge.weight.detach().norm()))
                        trace[arm].append(event);print(json.dumps(event),flush=True)
                del opt
                torch.save({"s1":{n:p.detach().cpu() for n,p in named.items()},"bridge":bridge.state_dict(),"feature_mode":mode},a.output/(arm+".pt"))
            evaluate(arm)
        handle.remove();mode="zero"
        # Separate Korean acoustic study: fixed EOS4 weights and fixed generated tokens.
        load("eos4")
        for r in ko:
            z,seed=generate(r);tok=hashlib.sha256(z.cpu().numpy().tobytes()).hexdigest()
            for arm in ARMS_KO:
                w,sr=synth(r,z,seed,arm)
                assert sr==48000
                save(r,arm,w,sr,token_sha256=tok,cap_hit=z.size(-1)>=512)
        assert len(rows)==72 and len(diagnostic)==128
        (a.output/"metrics.json").write_text(json.dumps(dict(status="completed",seed=a.seed,rows=rows,diagnostic=diagnostic,
            training_trace=trace,seconds=time.monotonic()-start),allow_nan=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true")
    for k in ("en-train","en-extra","en-val","en-test","ko-train","ko-test","weights","bert-model","output"):p.add_argument("--"+k,type=Path)
    p.add_argument("--seed",type=int,default=20261127);p.add_argument("--steps",type=int,default=240);a=p.parse_args()
    if a.selftest:selftest()
    else:main(a)
