"""v23 multi-hypothesis S1/S2 screening, experimental only."""
import argparse,hashlib,json,tempfile,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tools.s1_semantic_robustness import token_metrics
ARMS_EN=("ar","ar_soft","ar_band","ar_exact","ar_greedy","ar_lowtemp","ar_norep","nar_one","nar_iter","pre_one","pre_iter","pre_local","pre_repair")
ARMS_KO=("cfg10","cfg13","cfg16","cfg19","sway05","sway10","euler64","mid16","gain13","trim13")
def selftest():
    orig=torch.nn.Linear(1024,8)
    bridge=torch.nn.Linear(768,8,bias=False)
    torch.nn.init.zeros_(bridge.weight)
    z=torch.zeros(1,4,1024);expected=orig(z).detach()
    h=orig.register_forward_hook(lambda m,args,out:m.bias+bridge(args[0][...,:768]))
    assert torch.equal(orig(z),expected)
    orig(torch.ones_like(z)).sum().backward()
    assert torch.count_nonzero(bridge.weight.grad)>0
    h.remove()
    from tools.s1_english_word_alignment import selftest as test_alignment
    test_alignment()
    from tools.s1_masked_semantic import selftest as test_masked
    test_masked()
    from tools.s1_pretrained_masked import selftest as test_pretrained
    test_pretrained()
    print("Masked semantic and multilingual bridge selftests passed")
def main(a):
    import soundfile as sf
    from TTS_infer_pack.TTS import TTS,TTS_Config,load_audio_gpu
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
        from tools.s1_english_word_alignment import variants
        coverage={}
        e.text_preprocessor.multilingual_bert=None
        assert all(p.dtype==torch.float32 for p in extractor.model.parameters())
        @torch.no_grad()
        def cache(folder,lang):
            m=json.loads((folder/"manifest.json").read_text());pool=[];pairs=[]
            frontend="en" if lang=="en" else "all_ko"
            def one(r):
                p=folder/r["path"];assert hashlib.sha256(p.read_bytes()).hexdigest()==r["pcm_sha256"]
                text=r["text"]
                if text[-1] not in ".!?。！？":text+="."
                phones,b,norm=e.text_preprocessor.segment_and_extract_feature_for_text(text,frontend,e.configs.version)
                x=torch.as_tensor(phones,device=device,dtype=torch.long)[None]
                b=torch.as_tensor(b,device=device,dtype=torch.float32)[None]
                assert b.shape==(1,1024,x.size(1)) and not torch.count_nonzero(b)
                if lang=="en":
                    bb,exact=variants(extractor,norm,phones,e.configs.version)
                    bb={"word":bb["word"][None].to(device)}
                    coverage.setdefault(str(folder.name),[0,0]);coverage[str(folder.name)][0]+=int(exact);coverage[str(folder.name)][1]+=1
                    assert all(v.shape==b.shape and torch.isfinite(v).all() for v in bb.values())
                else:bb={"word":b}
                w,_=load_audio_gpu(str(p),16000,device,mono=True,dtype=torch.float32,match_librosa=True)
                h=e.cnhuhbert_model.model(w.reshape(1,-1))["last_hidden_state"].transpose(1,2)
                y=e.vits_model.extract_latent(h)[0,0].reshape(1,-1).long()
                return dict(x=x,b=b,bb=bb,y=y,path=p,lang=lang,start=0,phones=x)
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
        assert sum(v[0] for v in coverage.values())>=.8*sum(v[1] for v in coverage.values()),coverage
        del extractor
        torch.cuda.empty_cache()
        enpool=et+extra_pairs;kopool=kt
        from tools.s1_masked_semantic import MaskedSemantic,mask_for,decode
        torch.manual_seed(a.seed)
        nar=MaskedSemantic(model.phoneme_vocab_size).to(device)
        nar.load_state_dict(torch.load(a.nar_weights/"masked-semantic.pt",map_location="cpu",weights_only=True));nar.eval()
        trace={"pretrained":[]}
        nar_parameters=sum(p.numel() for p in nar.parameters())
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
                q["b"]=torch.cat(tuple(p["b"] if mode=="zero" else p["bb"][mode] for p in (r["ref"],r["target"])),2)
            else:q["b"]=r["b"] if mode=="zero" else r["bb"][mode]
            return q
        rows=[];diagnostic=[]
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
        def generate(r,arm="ar",expected=0,return_conf=False):
            seed=a.seed+r["ri"]*1009+r["ti"];torch.manual_seed(seed)
            from tools.s1_pretrained_masked import duration_logits
            control={"ar_soft":"soft","ar_band":"band","ar_exact":"exact"}.get(arm,"native")
            step=0;prob=[]
            def bias(module,inp,out):
                nonlocal step
                z=duration_logits(out,step,expected,control,model.EOS);step+=1;return z
            h=model.ar_predict_layer.register_forward_hook(bias)
            try:
                y,idx=next(model.infer_panel_naive(r["x"],torch.tensor([r["x"].size(1)],device=device),r["prompt"],conditioned(r)["b"],
                    top_k=1 if arm=="ar_greedy" else 15,top_p=1.,early_stop_num=511,temperature=.7 if arm=="ar_lowtemp" else 1.,
                    repetition_penalty=1. if arm=="ar_norep" else 1.35,probability_trace=prob if return_conf else None))
            finally:h.remove()
            assert torch.equal(y[:,:r["start"]],r["prompt"])
            z=y[:,r["start"]:].reshape(1,1,-1).long()
            assert 0<z.size(-1)<=512 and z.max()<1024
            if return_conf:
                assert len(prob)>=z.size(-1)
                conf=torch.stack([p.reshape(-1,p.size(-1))[0,int(tok)] for p,tok in zip(prob,z.flatten())])[None]
                return z,seed,conf
            return z,seed
        @torch.no_grad()
        def synth(r,z,seed,acoustic="cfg13"):
            refpath=r["ref"]["path"];rp=r["prompt"]
            if r["lang"]=="ko" and acoustic=="gain13":
                w,rs=sf.read(refpath,dtype="float32");w,_=level_match(w,target_db=-23.,max_gain=8.)
                refpath=tmp/"gain.wav";sf.write(refpath,w,rs,subtype="FLOAT")
            if r["lang"]=="ko" and acoustic=="trim13":rp=rp[:,:r["ref"]["y"].size(1)]
            e.set_ref_audio(str(refpath))
            e.prompt_cache["prompt_semantic"]=rp.flatten().clone()
            e.prompt_cache["phones"]=r["ref"]["x"][0].tolist()
            cfm=e.vits_model.cfm;original=cfm.inference
            cfg_rate={"cfg10":1.,"cfg13":1.3,"cfg16":1.6,"cfg19":1.9}.get(acoustic,1.3)
            steps=64 if acoustic=="euler64" else 16 if acoustic=="mid16" else 32
            sway={"sway05":-.5,"sway10":-1.}.get(acoustic,0.)
            try:
                if r["lang"]=="ko":
                    cfm.inference=lambda *args,**kw:research_v5(cfm,*args,**kw,method="midpoint" if acoustic=="mid16" else "euler",sway=sway,noise_override=.875)
                torch.manual_seed(seed+100000)
                wav=e.using_vocoder_synthesis(z,r["target"]["x"],sample_steps=steps,cfg_rate=cfg_rate if r["lang"]=="ko" else .9)
                return wav.detach().float().cpu().numpy(),int(e.vocoder_configs["sr"])
            finally:cfm.inference=original
        # Recover v21 word BERT AR checkpoint; compare same corpus, not matched parameter budget.
        state=torch.load(a.ar_weights/"word.pt",map_location="cpu",weights_only=True)
        assert set(state["s1"])==set(named) and state["feature_mode"]=="word"
        with torch.no_grad():
            for n,p in named.items():p.copy_(state["s1"][n])
            bridge.load_state_dict(state["bridge"])
        mode="word"
        from tools.s1_pretrained_masked import PretrainedMasked,fill,low_conf_mask
        pre=PretrainedMasked(model,lambda r:conditioned(r)["b"])
        # Score AR and original NAR first; keep repair seeds for post-training comparison.
        pending=[]
        with torch.no_grad():
            for split,rr in (("validation",val),("test",test)):
                for r in rr:
                    y=r["target"]["y"];n=nar.length(r);seed=a.seed+r["ri"]*1009+r["ti"]
                    lg,ll=nar(r,torch.full_like(y,1024))
                    diagnostic.append(dict(split=split,speaker=r["ri"],target=r["ti"],true_length=y.size(1),predicted_length=n,
                        full_mask_ce=float(F.cross_entropy(lg.reshape(-1,1024),y.flatten())),
                        full_mask_top1=float((lg.argmax(-1)==y).float().mean())))
                    if split!="test" or r["ti"] not in (1,4):continue
                    def timed(fn):
                        torch.cuda.synchronize();t=time.monotonic();v=fn();torch.cuda.synchronize()
                        return v,time.monotonic()-t
                    (z,seed,conf),ar_seconds=timed(lambda:generate(r,return_conf=True))
                    pending.append((r,z,seed,conf,ar_seconds,n))
                    options={"ar":(z.flatten()[None],ar_seconds),
                        "nar_one":timed(lambda:decode(nar,r,n,1,seed)),
                        "nar_iter":timed(lambda:decode(nar,r,n,8,seed))}
                    for arm in ("ar_soft","ar_band","ar_exact","ar_greedy","ar_lowtemp","ar_norep"):
                        (q,_),elapsed=timed(lambda:generate(r,arm,n))
                        options[arm]=(q.flatten()[None],elapsed)
                    for arm,(q,elapsed) in options.items():
                        w,sr=synth(r,q.reshape(1,1,-1),seed);assert sr==48000
                        save(r,arm,w,sr,cap_hit=q.size(1)>=512,generation_seconds=elapsed,
                            **token_metrics(y.flatten().cpu().numpy(),q.flatten().cpu().numpy()))
        # Adapt only native last-eight layers and output/text embeddings (same named set as v21).
        rng=np.random.default_rng(a.seed+7000);g=torch.Generator(device=device).manual_seed(a.seed+7001)
        params=list(named.values())+list(bridge.parameters())
        opt=torch.optim.AdamW(params,lr=2e-5,weight_decay=.01)
        for step in range(1,a.steps+1):
            pool=kopool if step%4==0 else enpool
            r=pool[int(rng.integers(len(pool)))];y=r["target"]["y"]
            masked,m=mask_for(y,1. if step%4==1 else float(rng.uniform(.2,1.)),g)
            lg=pre(r,masked,local=bool(step%2));loss=F.cross_entropy(lg[m],y[m])
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
            if step==1 or step%40==0 or step==a.steps:
                event=dict(arm="pre_iter",status="training",reference_index=step,text_index=a.steps,ce=float(loss.detach()))
                trace["pretrained"].append(event);print(json.dumps(event),flush=True)
        del opt
        torch.save({"s1":{n:p.detach().cpu() for n,p in named.items()},"bridge":bridge.state_dict()},a.output/"pretrained-masked.pt")
        with torch.no_grad():
            for r,z,seed,conf,ar_seconds,n in pending:
                y=r["target"]["y"]
                blank=torch.full((1,n),1024,device=device,dtype=torch.long)
                options={"pre_one":timed(lambda:fill(lambda t:pre(r,t),blank,1)),
                         "pre_iter":timed(lambda:fill(lambda t:pre(r,t),blank,8)),
                         "pre_local":timed(lambda:fill(lambda t:pre(r,t,local=True),blank,8)),
                         "pre_repair":timed(lambda:fill(lambda t:pre(r,t),low_conf_mask(z.flatten()[None],conf),8))}
                for arm,(q,elapsed) in options.items():
                    w,sr=synth(r,q.reshape(1,1,-1),seed);assert sr==48000
                    save(r,arm,w,sr,cap_hit=q.size(1)>=512,generation_seconds=elapsed+(ar_seconds if arm=="pre_repair" else 0.),
                        **token_metrics(y.flatten().cpu().numpy(),q.flatten().cpu().numpy()))
        handle.remove();mode="zero"
        # Separate Korean acoustic study: fixed EOS4 weights and fixed generated tokens.
        load("eos4")
        for r in ko:
            z,seed=generate(r);tok=hashlib.sha256(z.cpu().numpy().tobytes()).hexdigest()
            for arm in ARMS_KO:
                w,sr=synth(r,z,seed,arm)
                assert sr==48000
                save(r,arm,w,sr,token_sha256=tok,cap_hit=z.size(-1)>=512)
        assert len(rows)==184 and len(diagnostic)==32
        (a.output/"metrics.json").write_text(json.dumps(dict(status="completed",seed=a.seed,rows=rows,diagnostic=diagnostic,
            training_trace=trace,alignment_coverage=coverage,nar_parameters=nar_parameters,seconds=time.monotonic()-start),allow_nan=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true")
    for k in ("en-train","en-extra","en-val","en-test","ko-train","ko-test","weights","ar-weights","nar-weights","bert-model","output"):p.add_argument("--"+k,type=Path)
    p.add_argument("--seed",type=int,default=20261127);p.add_argument("--steps",type=int,default=400);a=p.parse_args()
    if a.selftest:selftest()
    else:main(a)
