"""Expanded multilingual S1 alignment pilot; no production modifications.
Two seeds, identical train batches, reset to same checkpoint before each arm.
"""
import argparse,hashlib,json,time,tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tools.s1_eos_root_cause import summary_logits
from tools.s1_semantic_robustness import token_metrics

ARMS={"eos2":(2.,0.),"eos4":(4.,0.),"eos8":(8.,0.),"eos4_ctc":(4.,.2),"eos4_ctc_strong":(4.,.6)}

def pack(rows,device,left):
    n=max(r["x"].size(-1) for r in rows);m=max(r["y"].size(-1) for r in rows)
    xl=torch.tensor([r["x"].size(-1) for r in rows],device=device)
    yl=torch.tensor([r["y"].size(-1) for r in rows],device=device)
    def px(t):
        d=n-t.size(-1);return F.pad(t,(d,0) if left else (0,d))
    return (torch.cat([px(r["x"]) for r in rows]),xl,
            torch.cat([F.pad(r["y"],(0,m-r["y"].size(-1)),value=1024) for r in rows]),yl,
            torch.cat([px(r["b"]) for r in rows]))
def loss_fn(logits,labels,lengths,mask_eos,weight,starts=None):
    pos=torch.arange(labels.size(1),device=labels.device)[None,:]
    valid=pos<=lengths[:,None]
    if starts is not None:valid=valid & (pos>=starts[:,None])
    eos=pos==lengths[:,None]
    w=(valid if mask_eos else torch.ones_like(valid)).to(logits.dtype)
    w=w*(1+(weight-1)*eos.to(w.dtype))
    ce=F.cross_entropy(logits.transpose(1,2),labels,reduction="none")
    return (ce*w).sum()/w.sum()
def alignment_loss(h,rows,head,n):
    # CTC uses only true target acoustic positions, excluding prompt and padding.
    values=[];targets=[];lens=[];tl=[]
    for i,r in enumerate(rows):
        phones=r.get("phones",r["x"]).flatten()
        p=int(r.get("start",0));length=r["y"].size(-1)-p
        repeats=int((phones[1:]==phones[:-1]).sum())
        if length<len(phones)+repeats:continue
        assert phones.min()>=0 and phones.max()<head.out_features-1
        values.append(head(h[i,n+p:n+p+length]).float())
        targets.append(phones);lens.append(length);tl.append(len(phones))
    if not values:return h.sum()*0,0
    # Padded scores are ignored by CTC lengths.
    padded=torch.nn.utils.rnn.pad_sequence(values).log_softmax(-1)
    loss=F.ctc_loss(padded,torch.cat(targets),torch.tensor(lens,dtype=torch.long),
        torch.tensor(tl,dtype=torch.long),blank=head.out_features-1,zero_infinity=False,reduction="mean")
    return loss,len(values)
def selftest():
    r=[dict(x=torch.tensor([[1,2,3]]),y=torch.tensor([[7,8,9]]),b=torch.ones(1,2,3)),
       dict(x=torch.tensor([[4,5,6,7,8]]),y=torch.tensor([[1,2,3,4,5]]),b=torch.ones(1,2,5))]
    x,xl,y,yl,b=pack(r,"cpu",True)
    assert x[0].tolist()==[0,0,1,2,3] and b[0,0].tolist()==[0,0,1,1,1]
    x,xl,y,yl,b=pack(r,"cpu",False)
    assert x[0].tolist()==[1,2,3,0,0]
    labels=torch.cat((y,torch.full((2,1),1024)),1)
    logits=torch.zeros(2,6,1025,requires_grad=True)
    loss_fn(logits,labels,yl,True,1.).backward()
    assert logits.grad[0,4:].abs().sum()==0 and logits.grad[0,3].abs().sum()>0
    h=torch.randn(2,10,8,requires_grad=True);head=torch.nn.Linear(8,12)
    loss,count=alignment_loss(h,r,head,5);assert count==2 and torch.isfinite(loss)
    loss.backward();assert torch.isfinite(h.grad).all()
    logits=torch.zeros(2,6,1025,requires_grad=True)
    loss_fn(logits,labels,yl,True,4.,torch.tensor([2,1])).backward()
    assert logits.grad[0,:2].abs().sum()==0 and logits.grad[0,3].abs().sum()>0
    print("Expanded alignment pilot selftest passed")

def main(a):
    from TTS_infer_pack.TTS import TTS,TTS_Config,load_audio_gpu
    from tools.evaluate_s1_next_waveforms import features
    from tools.s1_waveform_frontend import ensure_worker_nltk
    from tools.prepare_acoustic_zeroth_pairs import normalize
    import statistics as st
    a.output.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    start=time.monotonic()
    folders=(a.train,a.validation,a.test)
    manifests=[json.loads((p/"manifest.json").read_text()) for p in folders]
    def items(m):return [i for p in m["pairs"] for i in [p["reference"],*p["targets"]]]
    for i,m in enumerate(manifests):
        assert m["sample_rate"]==48000 and m["revision"]==manifests[0]["revision"]
        for j in range(i):
            assert not {p["speaker_id"] for p in m["pairs"]}&{p["speaker_id"] for p in manifests[j]["pairs"]}
            assert not {normalize(r["text"]) for r in items(m)}&{normalize(r["text"]) for r in items(manifests[j])}
            assert not {r["sha256"] for r in items(m)}&{r["sha256"] for r in items(manifests[j])}
    with tempfile.TemporaryDirectory() as td:
        cfg=TTS_Config({"custom":dict(TTS_Config.default_configs["v5dev"],device="cuda:0",is_half=False)})
        cfg.configs_path=str(Path(td)/"tts.yaml");e=TTS(cfg);ensure_worker_nltk()
        assert e.precision==torch.float32
        model=e.t2s_model.model.eval();device=e.configs.device;assert model.EOS==1024
        @torch.no_grad()
        def cache(folder,m,language="en"):
            rows=[];pairs=[]
            for si,p in enumerate(m["pairs"],1):
                def one(r):
                    path=folder/r["path"];assert hashlib.sha256(path.read_bytes()).hexdigest()==r["pcm_sha256"]
                    x,b=features(e,r["text"],language)
                    w,_=load_audio_gpu(str(path),16000,device,mono=True,dtype=torch.float32,match_librosa=True)
                    h=e.cnhuhbert_model.model(w.reshape(1,-1))["last_hidden_state"].transpose(1,2)
                    y=e.vits_model.extract_latent(h)[0,0].reshape(1,-1).long()
                    assert y.max()<1024
                    return dict(x=x,b=b,y=y,path=path)
                ref=one(p["reference"]);rows.append(ref)
                e.set_ref_audio(str(ref["path"]));prompt=e.prompt_cache["prompt_semantic"].reshape(1,-1).long().clone()
                for ti,t in enumerate(p["targets"],1):
                    r=one(t);rows.append(r)
                    pairs.append(dict(speaker=si,target=ti,x=torch.cat((ref["x"],r["x"]),1),
                        b=torch.cat((ref["b"],r["b"]),2),prompt=prompt,y=r["y"],single=r))
            return rows,pairs
        train,trainpairs=cache(folders[0],manifests[0]);_,val=cache(folders[1],manifests[1]);_,test=cache(folders[2],manifests[2])
        assert len(train)==80 and len(val)==16 and len(test)==16
        km=json.loads((a.ko/"manifest.json").read_text());assert km["sample_rate"]==16000
        _,ko=cache(a.ko,km,"all_ko");assert len(ko)==4
        kt=json.loads((a.ko_train/"manifest.json").read_text())
        assert kt["sample_rate"]==16000 and kt["revision"]==km["revision"]
        assert not {str(p["speaker_id"]) for p in kt["pairs"]}&{str(p["speaker_id"]) for p in km["pairs"]}
        assert not {normalize(r["text"]) for r in items(kt)}&{normalize(r["text"]) for r in items(km)}
        assert not {r["sha256"] for r in items(kt)}&{r["sha256"] for r in items(km)}
        ktrain,kpairs=cache(a.ko_train,kt,"all_ko");assert len(ktrain)==32
        def prompted(rows):
            return [dict(x=r["x"],b=r["b"],y=torch.cat((r["prompt"],r["y"]),1),
                start=r["prompt"].size(1),phones=r["single"]["x"]) for r in rows]
        en_pool=[*train,*prompted(trainpairs)];ko_pool=[*ktrain,*prompted(kpairs)]
        # Eight decoder blocks and text embedding, shared scope across all arms.
        # Retain phoneme vocabulary/checkpoint compatibility; auxiliary head discarded at inference.
        for p in model.parameters():p.requires_grad_(False)
        for module in (*model.h.layers[-8:],model.ar_text_embedding,model.ar_predict_layer):
            for p in module.parameters():p.requires_grad_(True)
        named={n:p for n,p in model.named_parameters() if p.requires_grad}
        assert named and any("ar_predict_layer" in n for n in named)
        initial={n:p.detach().cpu().clone() for n,p in named.items()}
        def reset():
            with torch.no_grad():
                for n,p in named.items():p.copy_(initial[n])
        def forward(x,xl,y,yl,b,return_hidden=False):
            xy,mask,labels=model.make_input_data(x,xl,y,yl,b)
            h,_=model.h((xy,None),mask=mask)
            logits=model.ar_predict_layer(h[:,x.size(1)-1:])
            return (logits,labels,h) if return_hidden else (logits,labels)
        @torch.no_grad()
        def evaluate(rows,arm,split,generate=False):
            results=[]
            for r in rows:
                x,b,prompt,y=[r[k] for k in ("x","b","prompt","y")]
                yy=torch.cat((prompt,y),1)
                lg,lbl=forward(x,torch.tensor([x.size(1)],device=device),yy,torch.tensor([yy.size(1)],device=device),b)
                lg=lg[:,prompt.size(1):];lbl=lbl[:,prompt.size(1):]
                assert torch.equal(lbl[:,:-1],y)
                out=dict(arm=arm,split=split,speaker=r["speaker"],target=r["target"],**summary_logits(lg,lbl,model.EOS))
                if generate:
                    torch.manual_seed(a.seed+r["speaker"]*1009+r["target"])
                    seq,idx=next(model.infer_panel_naive(x,torch.tensor([x.size(1)],device=device),prompt,b,
                        top_k=15,top_p=1.,early_stop_num=511,temperature=1.,repetition_penalty=1.35))
                    assert torch.equal(seq[:,:prompt.size(1)],prompt)
                    z=seq[:,prompt.size(1):].flatten().cpu().numpy()
                    assert 0<len(z)<=512 and z.max()<1024
                    out.update(cap_hit=len(z)>=512,**token_metrics(y.flatten().cpu().numpy(),z))
                results.append(out)
                print(json.dumps(dict(arm=arm,status=split,reference_index=r["speaker"],text_index=r["target"])),flush=True)
            return results
        rng=np.random.default_rng(a.seed)
        batches=[(rng.choice(len(en_pool),size=3,replace=False).tolist(),int(rng.integers(len(ko_pool)))) for _ in range(a.steps)]
        results=[];timings={};coverage={};trainlog={}
        results.extend(evaluate(val,"frozen","validation"))
        results.extend(evaluate(test,"frozen","test",True))
        results.extend(evaluate(ko,"frozen","ko_guard",True))
        for arm,(weight,ctc_weight) in ARMS.items():
            reset();torch.manual_seed(a.seed);t=time.monotonic()
            head=torch.nn.Linear(model.model_dim,model.phoneme_vocab_size+1).to(device)
            params=list(named.values())+(list(head.parameters()) if ctc_weight else [])
            opt=torch.optim.AdamW(params,lr=2e-5,weight_decay=.01)
            count_total=0;sample_total=0;trainlog[arm]=[]
            # Eval mode disables dropout across arms; gradients remain enabled.
            model.eval()
            for step,(idx,ki) in enumerate(batches,1):
                rows=[en_pool[i] for i in idx]+[ko_pool[ki]]
                x,xl,y,yl,b=pack(rows,device,True)
                starts=torch.tensor([r.get("start",0) for r in rows],device=device)
                opt.zero_grad(set_to_none=True)
                lg,labels,h=forward(x,xl,y,yl,b,True)
                ce=loss_fn(lg,labels,yl,True,weight,starts)
                if ctc_weight:
                    ctc,count=alignment_loss(h,rows,head,x.size(1));count_total+=count;sample_total+=len(rows)
                else:ctc=ce.detach()*0
                loss=ce+ctc_weight*ctc
                assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                opt.step()
                if step==1 or step%20==0 or step==a.steps:
                    event=dict(arm=arm,status="training",reference_index=step,text_index=a.steps,loss=float(loss.detach()),ce=float(ce.detach()),ctc=float(ctc.detach()))
                    trainlog[arm].append(event);print(json.dumps(event),flush=True)
            coverage[arm]=dict(eligible=count_total,total=sample_total)
            if ctc_weight:assert count_total>0
            del opt,head,params
            assert any(not torch.equal(p.detach().cpu(),initial[n]) for n,p in named.items())
            results.extend(evaluate(val,arm,"validation"))
            results.extend(evaluate(test,arm,"test",True))
            results.extend(evaluate(ko,arm,"ko_guard",True))
            torch.save({n:p.detach().cpu() for n,p in named.items()},a.output/(arm+".pt"))
            timings[arm]=time.monotonic()-t
        report=dict(status="completed",seed=a.seed,updates_per_arm=a.steps,train_utterances=len(train)+len(ktrain),en_train=len(train),ko_train=len(ktrain),ctc_coverage=coverage,training_trace=trainlog,
            trainable=sum(p.numel() for p in named.values()),arms=list(ARMS),rows=results,timings=timings,seconds=time.monotonic()-start,summary={})
        for arm in ("frozen",*ARMS):
            for split in ("validation","test","ko_guard"):
                rr=[r for r in results if r["arm"]==arm and r["split"]==split]
                agg=dict(cases=len(rr),median_eos_prob=st.median(r["eos_prob"] for r in rr),
                    median_eos_rank=st.median(r["eos_rank"] for r in rr),nll=st.mean(r["nll"] for r in rr),
                    top1=st.mean(r["token_top1"] for r in rr),early_eos_max_median=st.median(r["early_max"] for r in rr))
                if split in ("test","ko_guard"):agg.update(cap_hits=sum(r["cap_hit"] for r in rr),ued=sum(r["unit_errors"] for r in rr)/sum(r["reference_tokens"] for r in rr),
                    median_abs_log_length_error=st.median(abs(np.log(r["length_ratio"])) for r in rr))
                report["summary"][arm+"-"+split]=agg
        (a.output/"metrics.json").write_text(json.dumps(report,allow_nan=False,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true")
    for k in ("train","validation","test","ko","ko-train","output"):p.add_argument("--"+k,type=Path)
    p.add_argument("--seed",type=int,default=20261127);p.add_argument("--steps",type=int,default=240);a=p.parse_args()
    if a.selftest:selftest()
    else:main(a)
