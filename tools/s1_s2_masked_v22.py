"""v22: masked semantic generator vs AR baseline and fixed-token CFM noise study."""
import argparse,hashlib,json,tempfile,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tools.s1_expanded_alignment_pilot import pack,loss_fn
from tools.s1_eos_root_cause import summary_logits
from tools.s1_semantic_robustness import token_metrics
ARMS_EN=("ar","nar_one","nar_iter","nar_oracle_len","nar_repair")
ARMS_KO=("noise0875","noise065","noise100","noise110")
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
    from tools.s1_english_word_alignment import selftest as test_alignment
    test_alignment()
    from tools.s1_masked_semantic import selftest as test_masked
    test_masked()
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
                    bb={k:v[None].to(device) for k,v in bb.items()}
                    coverage.setdefault(str(folder.name),[0,0]);coverage[str(folder.name)][0]+=int(exact);coverage[str(folder.name)][1]+=1
                    assert all(v.shape==b.shape and torch.isfinite(v).all() for v in bb.values())
                else:bb={k:b for k in ("nearest","word","pooled")}
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
        opt=torch.optim.AdamW(nar.parameters(),lr=3e-4,weight_decay=.01)
        rng=np.random.default_rng(a.seed+4000)
        g=torch.Generator(device=device).manual_seed(a.seed+5000)
        trace={"masked":[]}
        for step in range(1,a.steps+1):
            pool=kopool if step%4==0 else enpool
            r=pool[int(rng.integers(len(pool)))];y=r["target"]["y"]
            ratio=1. if step%4==1 else float(rng.uniform(.15,1.))
            masked,m=mask_for(y,ratio,g)
            lg,ll=nar(r,masked)
            ce=F.cross_entropy(lg[m],y[m]);dur=F.smooth_l1_loss(ll,torch.full_like(ll,np.log(y.size(1))))
            loss=ce+.5*dur
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(nar.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step==1 or step%100==0 or step==a.steps:
                event=dict(arm="nar_iter",status="training",reference_index=step,text_index=a.steps,ce=float(ce.detach()),duration_loss=float(dur.detach()))
                trace["masked"].append(event);print(json.dumps(event),flush=True)
        nar.eval();del opt
        torch.save(nar.state_dict(),a.output/"masked-semantic.pt")
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
        def forward(rows):
            x,xl,y,yl,b=pack([conditioned(r) for r in rows],device,True)
            xy,mask,labels=model.make_input_data(x,xl,y,yl,b)
            h,_=model.h((xy,None),mask=mask)
            lg=model.ar_predict_layer(h[:,x.size(1)-1:])
            starts=torch.tensor([r["start"] for r in rows],device=device)
            return lg,labels,starts,yl
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
        def generate(r):
            seed=a.seed+r["ri"]*1009+r["ti"];torch.manual_seed(seed)
            y,idx=next(model.infer_panel_naive(r["x"],torch.tensor([r["x"].size(1)],device=device),r["prompt"],conditioned(r)["b"],
                top_k=15,top_p=1.,early_stop_num=511,temperature=1.,repetition_penalty=1.35))
            assert torch.equal(y[:,:r["start"]],r["prompt"])
            z=y[:,r["start"]:].reshape(1,1,-1).long()
            assert 0<z.size(-1)<=512 and z.max()<1024
            return z,seed
        @torch.no_grad()
        def synth(r,z,seed,acoustic="noise0875",reference_prompt=None):
            e.set_ref_audio(str(r["ref"]["path"]))
            # Keep semantic conditioning identical even when acoustic reference gain changes.
            e.prompt_cache["prompt_semantic"]=(r["prompt"] if reference_prompt is None else reference_prompt).flatten().clone()
            e.prompt_cache["phones"]=r["ref"]["x"][0].tolist()
            cfm=e.vits_model.cfm;original=cfm.inference
            scales={"noise0875":.875,"noise065":.65,"noise100":1.,"noise110":1.1}
            try:
                if r["lang"]=="ko" and acoustic!="noise0875":
                    cfm.inference=lambda *args,**kw:research_v5(cfm,*args,**kw,method="euler",noise_override=scales[acoustic])
                torch.manual_seed(seed+100000)
                wav=e.using_vocoder_synthesis(z,r["target"]["x"],sample_steps=32,cfg_rate=1.3 if r["lang"]=="ko" else .9)
                return wav.detach().float().cpu().numpy(),int(e.vocoder_configs["sr"])
            finally:cfm.inference=original
        # Recover v21 word BERT AR checkpoint; compare same corpus, not matched parameter budget.
        state=torch.load(a.ar_weights/"word.pt",map_location="cpu",weights_only=True)
        assert set(state["s1"])==set(named) and state["feature_mode"]=="word"
        with torch.no_grad():
            for n,p in named.items():p.copy_(state["s1"][n])
            bridge.load_state_dict(state["bridge"])
        mode="word"
        with torch.no_grad():
            for split,rr in (("validation",val),("test",test)):
                for r in rr:
                    y=r["target"]["y"];n=nar.length(r);seed=a.seed+r["ri"]*1009+r["ti"]
                    lg,ll=nar(r,torch.full_like(y,1024))
                    diagnostic.append(dict(split=split,speaker=r["ri"],target=r["ti"],true_length=y.size(1),predicted_length=n,
                        full_mask_ce=float(F.cross_entropy(lg.reshape(-1,1024),y.flatten())),
                        full_mask_top1=float((lg.argmax(-1)==y).float().mean())))
                    if split!="test" or r["ti"] not in (1,4):continue
                    z,seed=generate(r)
                    options={"ar":z.flatten()[None],"nar_one":decode(nar,r,n,1,seed),
                        "nar_iter":decode(nar,r,n,8,seed),"nar_oracle_len":decode(nar,r,y.size(1),8,seed),
                        "nar_repair":decode(nar,r,z.size(-1),8,seed,initial=z.flatten()[None],repair_ratio=.30)}
                    for arm,q in options.items():
                        w,sr=synth(r,q.reshape(1,1,-1),seed);assert sr==48000
                        save(r,arm,w,sr,cap_hit=q.size(1)>=512,oracle_length=arm=="nar_oracle_len",
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
        assert len(rows)==72 and len(diagnostic)==32
        (a.output/"metrics.json").write_text(json.dumps(dict(status="completed",seed=a.seed,rows=rows,diagnostic=diagnostic,
            training_trace=trace,alignment_coverage=coverage,seconds=time.monotonic()-start),allow_nan=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true")
    for k in ("en-train","en-extra","en-val","en-test","ko-train","ko-test","weights","ar-weights","bert-model","output"):p.add_argument("--"+k,type=Path)
    p.add_argument("--seed",type=int,default=20261127);p.add_argument("--steps",type=int,default=1600);a=p.parse_args()
    if a.selftest:selftest()
    else:main(a)
