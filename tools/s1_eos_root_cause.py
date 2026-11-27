"""Cross-version EOS root-cause audit. No model training or production changes."""
import argparse,json,hashlib,time,tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tools.s1_semantic_robustness import token_metrics

def summary_logits(lg,labels,eos):
    assert lg.shape[:2]==labels.shape and int(labels[0,-1])==eos
    odds=lg[...,eos]-torch.logsumexp(lg[...,:eos],-1)
    rank=(lg[...,:eos]>lg[...,eos,None]).sum(-1)+1
    return dict(eos_prob=float(odds[0,-1].sigmoid()),eos_rank=int(rank[0,-1]),
        early_max=float(odds[0,:-1].sigmoid().max()),nll=float(F.cross_entropy(lg[0,:-1],labels[0,:-1])),
        token_top1=float((lg[0,:-1].argmax(-1)==labels[0,:-1]).float().mean()))
def selftest():
    l=torch.zeros(1,4,5);labels=torch.tensor([[0,1,2,4]]);l[0,-1,4]=10
    r=summary_logits(l,labels,4);assert r["eos_rank"]==1 and r["eos_prob"]>.99
    print("EOS diagnostic selftest passed")
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

@torch.inference_mode()
def main(a):
    import librosa
    from tools.my_utils import load_audio
    from TTS_infer_pack.TTS import TTS,TTS_Config,load_audio_gpu
    from tools.evaluate_s1_next_waveforms import features
    from tools.s1_waveform_frontend import ensure_worker_nltk
    a.output.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    start=time.monotonic()
    with tempfile.TemporaryDirectory() as td:
        cfg=TTS_Config({"custom":dict(TTS_Config.default_configs[a.version],device="cuda:0",is_half=False)})
        cfg.configs_path=str(Path(td)/"tts.yaml");e=TTS(cfg);ensure_worker_nltk()
        assert e.precision==torch.float32 and e.configs.version==a.version
        teacher=e.t2s_model.model.eval();v=e.vits_model.eval();device=e.configs.device
        assert teacher.EOS==1024
        report=dict(status="completed",version=a.version,checkpoint=str(e.configs.t2s_weights_path),
            s1_sha=hashlib.file_digest(open(e.configs.t2s_weights_path,"rb"),"sha256").hexdigest(),
            codebook_sha=digest(v.quantizer.vq.layers[0]._codebook.embed),
            projection_sha=digest(v.ssl_proj.weight),sample_rate=e.configs.sampling_rate,
            native_prompt_padding_samples=int(e.configs.sampling_rate*.3),rows=[],generation=[],tokens={})
        def audio(path):
            w,_=load_audio_gpu(str(path),16000,device,mono=True,dtype=torch.float32,match_librosa=True)
            return w.reshape(1,-1)
        def train_audio(path):
            w=load_audio(str(path),32000);peak=np.abs(w).max();assert peak>0
            w=librosa.resample(w*(1145.14*(.475/peak+.5)),orig_sr=32000,target_sr=16000)
            return torch.as_tensor(w,device=device,dtype=torch.float32).reshape(1,-1)
        def encode(w):
            h=e.cnhuhbert_model.model(w)["last_hidden_state"].transpose(1,2)
            return v.extract_latent(h)[0,0].reshape(1,-1).long()
        def forced(x,b,prompt,ids):
            y=torch.cat((prompt,ids),1);prefix=prompt.size(1)
            xy,mask,targets=teacher.make_input_data(x,torch.tensor([x.size(1)],device=device),y,torch.tensor([y.size(1)],device=device),b)
            h,_=teacher.h((xy,None),mask=mask)
            lg=teacher.ar_predict_layer(h[:,x.size(1)-1:])[:,prefix:].float()
            labels=targets[:,prefix:]
            assert torch.equal(labels[:,:-1],ids) and lg.size(1)==ids.size(1)+1
            # Run the same full prefix through actual native cached prefill kernel.
            n=x.size(1);m=y.size(1)
            xmask=F.pad(torch.zeros(n,n,dtype=torch.bool,device=device),(0,m),value=True)
            ymask=F.pad(torch.triu(torch.ones(m,m,dtype=torch.bool,device=device),diagonal=1),(n,0),value=False)
            native_mask=torch.cat((xmask,ymask),0)[None,None].expand(1,teacher.num_head,-1,-1)
            hh,_,_=teacher.t2s_transformer.process_prompt(xy,native_mask,None)
            native=teacher.ar_predict_layer(hh[:,-1]).float()
            end=lg[:,-1]
            r=summary_logits(lg,labels,teacher.EOS)
            r.update(native_last_max_abs=float((native-end).abs().max()),
                native_last_eos_prob=float(native.softmax(-1)[0,teacher.EOS]),
                native_last_top1_equal=bool(native.argmax(-1).item()==end.argmax(-1).item()))
            return r
        for folder,lang in ((a.en,"en"),(a.ko,"all_ko")):
            manifest=json.loads((folder/"manifest.json").read_text())
            assert manifest["verified_same_speaker"]
            for ri,pair in enumerate(manifest["pairs"],1):
                ref=pair["reference"];rp=folder/ref["path"]
                assert hashlib.sha256(rp.read_bytes()).hexdigest()==ref["pcm_sha256"]
                px,pb=features(e,ref["text"],lang)
                rw=audio(rp);plain=encode(rw);pad03=encode(F.pad(rw,(0,4800)));train_ref=encode(train_audio(rp))
                e.set_ref_audio(str(rp))
                native=e.prompt_cache["prompt_semantic"].reshape(1,-1).long().clone()
                reconstructed=encode(F.pad(rw,(0,int(e.configs.sampling_rate*.3))))
                assert torch.equal(native,reconstructed)
                print(json.dumps(dict(arm="reference_cached",status="ok",reference_index=ri,language=lang)),flush=True)
                for ti,t in enumerate(pair["targets"],1):
                    path=folder/t["path"]
                    assert hashlib.sha256(path.read_bytes()).hexdigest()==t["pcm_sha256"]
                    tx,tb=features(e,t["text"],lang);tw=audio(path);ids=encode(tw);train_ids=encode(train_audio(path))
                    key=f"{lang}-{ri}-{ti}"
                    report["tokens"][key]={"target":ids.flatten().cpu().tolist(),"plain":plain.flatten().cpu().tolist(),
                        "native":native.flatten().cpu().tolist(),"pad03":pad03.flatten().cpu().tolist(),
                        "train_target":train_ids.flatten().cpu().tolist(),"train_ref":train_ref.flatten().cpu().tolist()}
                    # Actual production target preprocessor versus prior research helper.
                    prod=e.text_preprocessor.preprocess(t["text"],lang,"cut0",e.configs.version)
                    prod_equal=len(prod)==1 and list(prod[0]["phones"])==tx[0].tolist()
                    x=torch.cat((px,tx),1);b=torch.cat((pb,tb),2)
                    for arm,xx,bb,p,yy in (("standalone",tx,tb,ids[:,:0],ids),("native_prompt",x,b,native,ids),
                            ("no_padding",x,b,plain,ids),("exact_300ms",x,b,pad03,ids),
                            ("train_standalone",tx,tb,ids[:,:0],train_ids),("train_concat",x,b,train_ref,train_ids)):
                        row=dict(language=lang,reference_index=ri,text_index=ti,arm=arm,status="ok",
                            target_tokens=yy.size(1),prompt_tokens=p.size(1),target_phones=tx.size(1),
                            production_phones_equal=prod_equal,bert_nonzero=int(torch.count_nonzero(tb)),
                            **forced(xx,bb,p,yy))
                        report["rows"].append(row)
                        print(json.dumps(row),flush=True)
                    # Free generation on first target of each speaker, two prompt variants.
                    if ti==1:
                        for arm,p in (("gen_native",native),("gen_no_padding",plain)):
                            torch.manual_seed(20261127+ri*1009)
                            y,idx=next(teacher.infer_panel_naive(x,torch.tensor([x.size(1)],device=device),p,b,
                                top_k=15,top_p=1.,early_stop_num=511,temperature=1.,repetition_penalty=1.35))
                            assert torch.equal(y[:,:p.size(1)],p)
                            seq=y[:,p.size(1):].flatten().cpu().numpy()
                            assert 0<len(seq)<=512 and seq.max()<1024
                            row=dict(language=lang,reference_index=ri,text_index=ti,arm=arm,status="ok",
                                cap_hit=len(seq)>=512,**token_metrics(ids.flatten().cpu().numpy(),seq))
                            report["generation"].append(row);print(json.dumps(row),flush=True)
                (a.output/"metrics.json").write_text(json.dumps(report,allow_nan=False,indent=2))
        report["seconds"]=time.monotonic()-start
        (a.output/"metrics.json").write_text(json.dumps(report,allow_nan=False,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true")
    for k in ("en","ko","output"):p.add_argument("--"+k,type=Path)
    p.add_argument("--version",choices=("v5dev","v2ProPlus"));a=p.parse_args()
    if a.selftest:selftest()
    else:main(a)
