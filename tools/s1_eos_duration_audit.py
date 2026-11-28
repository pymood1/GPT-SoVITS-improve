"""Frozen S1 EOS calibration research; train-only duration fit, independent evaluation.
No production change, no oracle duration at inference.
"""
import argparse,json,hashlib,time,tempfile
from pathlib import Path
import numpy as np
import torch
from tools.s1_semantic_robustness import token_metrics

def design(ref_phones,ref_tokens,target_phones):
    if min(ref_phones,ref_tokens,target_phones)<1:raise ValueError("empty input")
    return np.array([1.,np.log(target_phones),np.log(ref_tokens/ref_phones)],dtype=np.float64)
def fit_duration(rows):
    x=np.stack([r[0] for r in rows]);y=np.log([r[1] for r in rows])
    # prior: target length approximately proportional to text length and reference rate
    prior=np.array([0.,1.,1.]);reg=np.diag([.1,2.,2.])
    return np.linalg.solve(x.T@x+reg,x.T@y+reg@prior)
def predict(w,x):
    return float(np.clip(np.exp(x@w),12.,480.))
def eos_bias(step,expected,strength):
    # Smooth prior: no forced stop, penalize early EOS, favor late EOS.
    return float(strength*np.tanh((step/expected-1.)*3.))
def selftest():
    rr=[(design(30,90,n),3*n) for n in (20,30,40,50)]
    w=fit_duration(rr)
    assert abs(predict(w,design(30,90,40))-120)<1e-5
    assert eos_bias(10,100,4)<0<eos_bias(200,100,4)
    assert eos_bias(100,100,4)==0 and abs(eos_bias(500,100,4))<=4
    print("Duration predictor and EOS prior selftests passed")

@torch.no_grad()
def main(a):
    import soundfile as sf
    from tools.s1_s2_acoustic_isolation import level_match
    from TTS_infer_pack.TTS import TTS,TTS_Config,load_audio_gpu
    from tools.evaluate_s1_next_waveforms import features
    from tools.s1_waveform_frontend import ensure_worker_nltk
    from tools.prepare_acoustic_zeroth_pairs import normalize
    manifests=[json.loads((p/"manifest.json").read_text()) for p in (a.train,a.validation,a.test)]
    def items(m):return [i for p in m["pairs"] for i in [p["reference"],*p["targets"]]]
    for i,m in enumerate(manifests):
        assert m["sample_rate"]==48000 and m["revision"]==manifests[0]["revision"]
        for j in range(i):
            assert not {p["speaker_id"] for p in m["pairs"]}&{p["speaker_id"] for p in manifests[j]["pairs"]}
            assert not {normalize(r["text"]) for r in items(m)}&{normalize(r["text"]) for r in items(manifests[j])}
            assert not {r["sha256"] for r in items(m)}&{r["sha256"] for r in items(manifests[j])}
    a.output.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    start=time.monotonic()
    with tempfile.TemporaryDirectory() as td:
        cfg=TTS_Config({"custom":dict(TTS_Config.default_configs["v5dev"],device="cuda:0",is_half=False)})
        cfg.configs_path=str(Path(td)/"tts.yaml");e=TTS(cfg);ensure_worker_nltk()
        assert e.precision==torch.float32
        teacher=e.t2s_model.model.eval();v=e.vits_model.eval();device=e.configs.device
        assert teacher.EOS==1024
        def check(folder,row):
            p=folder/row["path"];assert hashlib.sha256(p.read_bytes()).hexdigest()==row["pcm_sha256"];return p
        def target_ids(path):
            w,_=load_audio_gpu(str(path),16000,device,mono=True,dtype=torch.float32,match_librosa=True)
            h=e.cnhuhbert_model.model(w.reshape(1,-1))["last_hidden_state"].transpose(1,2)
            return v.extract_latent(h)[0,0].reshape(1,-1).long()
        def cache(folder,m):
            rows=[]
            for si,pair in enumerate(m["pairs"]):
                ref=pair["reference"];e.set_ref_audio(str(check(folder,ref)))
                prompt=e.prompt_cache["prompt_semantic"].reshape(1,-1).long().clone()
                px,pb=features(e,ref["text"],"en")
                for j,t in enumerate(pair["targets"]):
                    tx,tb=features(e,t["text"],"en")
                    ids=target_ids(check(folder,t))
                    rows.append(dict(speaker=si+1,target=j+1,design=design(px.size(1),prompt.size(1),tx.size(1)),
                        prompt=prompt,x=torch.cat((px,tx),1),b=torch.cat((pb,tb),2),ids=ids,
                        ref_path=check(folder,ref),px=px,tx=tx))
            return rows
        tr=cache(a.train,manifests[0]);va=cache(a.validation,manifests[1]);te=cache(a.test,manifests[2])
        w=fit_duration([(r["design"],r["ids"].size(1)) for r in tr])
        def duration_report(rows):
            ratios=[predict(w,r["design"])/r["ids"].size(1) for r in rows]
            return dict(cases=len(rows),median_ratio=float(np.median(ratios)),median_abs_log_error=float(np.median(np.abs(np.log(ratios)))))
        def forced(r):
            x,b,prompt,ids=[r[k] for k in ("x","b","prompt","ids")]
            y=torch.cat((prompt,ids),1)
            xy,mask,targets=teacher.make_input_data(x,torch.tensor([x.size(1)],device=device),y,torch.tensor([y.size(1)],device=device),b)
            h,_=teacher.h((xy,None),mask=mask)
            lg=teacher.ar_predict_layer(h[:,x.size(1)-1:])[:,prompt.size(1):].float()
            labels=targets[:,prompt.size(1):]
            assert torch.equal(labels[:,:-1],ids) and int(labels[0,-1])==teacher.EOS and lg.size(1)==ids.size(1)+1
            eos=lg[...,teacher.EOS]
            odds=eos-torch.logsumexp(lg[...,:teacher.EOS],-1)
            rank=(lg[...,:teacher.EOS]>eos[...,None]).sum(-1)+1
            return dict(final_eos_prob=float(odds[0,-1].sigmoid()),final_eos_rank=int(rank[0,-1]),
                        max_early_eos_prob=float(odds[0,:-1].sigmoid().max()),
                        target_token_nll=float(torch.nn.functional.cross_entropy(lg[0,:-1],ids[0])),
                        final_eos_logodds=float(odds[0,-1]))
        def generate(r,strength,seed):
            x,b,prompt,ids=[r[k] for k in ("x","b","prompt","ids")]
            expected=predict(w,r["design"]);trace=[]
            def hook(module,inp,out):
                step=len(trace)
                raw=out.float()
                trace.append(float((raw[0,teacher.EOS]-torch.logsumexp(raw[0,:teacher.EOS],0)).sigmoid()))
                if strength==0:return None
                changed=out.clone();changed[...,teacher.EOS]+=eos_bias(step,expected,strength)
                return changed
            h=teacher.ar_predict_layer.register_forward_hook(hook)
            try:
                torch.manual_seed(seed)
                y,idx=next(teacher.infer_panel_naive(x,torch.tensor([x.size(1)],device=device),prompt,b,
                    top_k=15,top_p=1.,early_stop_num=511,temperature=1.,repetition_penalty=1.35))
            finally:h.remove()
            assert torch.equal(y[:,:prompt.size(1)],prompt)
            seq=y[:,prompt.size(1):].reshape(-1).cpu().numpy()
            assert 0<len(seq)<=512
            if a.synthesize:
                e.set_ref_audio(str(r["ref_path"]));e.prompt_cache["phones"]=r["px"][0].tolist()
                assert torch.equal(e.prompt_cache["prompt_semantic"].reshape(1,-1).long(),prompt)
                torch.manual_seed(seed+100000)
                wav=e.using_vocoder_synthesis(torch.as_tensor(seq,device=device,dtype=torch.long).reshape(1,1,-1),r["tx"],sample_steps=32,cfg_rate=.90)
                wav=wav.detach().float().cpu().numpy().reshape(-1)
                assert np.isfinite(wav).all() and len(wav)>=24000 and int(e.vocoder_configs["sr"])==48000
                wav,_=level_match(wav,max_gain=32.)
                sf.write(a.output/f"speaker-{r['speaker']:02}-target-{r['target']:02}-bias-{int(strength)}.wav",wav,48000,subtype="PCM_16")
            return dict(expected_tokens=expected,generated_tokens=len(seq),cap_hit=len(seq)>=512,
                final_raw_eos_prob=trace[-1],max_raw_eos_prob=max(trace),native_stop_idx=int(idx),
                **token_metrics(ids.flatten().cpu().numpy(),seq))
        result=dict(status="completed",seed=a.seed,duration_coefficients=w.tolist(),
                    validation_duration=duration_report(va),test_duration=duration_report(te),forced=[],rows=[])
        for r in te:
            result["forced"].append(dict(speaker=r["speaker"],target=r["target"],**forced(r)))
            seed=a.seed+r["speaker"]*1009+r["target"]
            for strength in (0.,2.,4.):
                result["rows"].append(dict(speaker=r["speaker"],target=r["target"],strength=strength,**generate(r,strength,seed)))
        result["summary"]={}
        for strength in (0.,2.,4.):
            rr=[r for r in result["rows"] if r["strength"]==strength]
            result["summary"][str(strength)]=dict(cases=len(rr),cap_hits=sum(r["cap_hit"] for r in rr),
                median_length_ratio=float(np.median([r["length_ratio"] for r in rr])),
                median_abs_log_length_error=float(np.median([abs(np.log(r["length_ratio"])) for r in rr])),
                ued=sum(r["unit_errors"] for r in rr)/sum(r["reference_tokens"] for r in rr))
        result["seconds"]=time.monotonic()-start
        result["caveat"]="Teacher-forcing diagnosis and token length are not speech quality. No real target length used in generation; English corpus small. Waveform evaluation optional and requires downstream scoring."
        (a.output/"metrics.json").write_text(json.dumps(result,allow_nan=False,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true")
    for k in ("train","validation","test","output"):p.add_argument("--"+k,type=Path)
    p.add_argument("--seed",type=int,default=20261127);p.add_argument("--synthesize",action="store_true");a=p.parse_args()
    if a.selftest:selftest()
    else:main(a)
