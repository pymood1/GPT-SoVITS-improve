"""Small independent bridge pilot using frozen original S1 and V5dev.

Teacher-forced reconstruction is a feasibility check, not free-running TTS
quality. Use public, preprocessed, recording-disjoint examples only.
"""
import argparse
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"GPT_SoVITS")]
os.environ.setdefault("version","v2")
os.environ.setdefault("hz","25hz")

import torch
from tools.train_s1_next import load_teacher,dataset,prepare
from AR.models.s1_next import SemanticResidualBridge
from module.s1_next_bridge import FrozenV5Bridge,paired_losses


@torch.no_grad()
def extract(teacher,cfg,data,indices,folder,device,allow_missing):
    from tools.my_utils import load_audio
    from module.mel_processing import spectrogram_torch,spec_to_mel_torch
    out=[]
    for idx in indices:
        x,b,y=prepare(data[idx],cfg,device,allow_missing)
        name=data.item_names[idx]
        xy,mask,_=teacher.make_input_data(x,torch.tensor([x.size(1)],device=device),
            y,torch.tensor([y.size(1)],device=device),b)
        hidden,_=teacher.h((xy,None),mask=mask)
        hidden=hidden[:,x.size(1):]
        if hidden.size(1)!=y.size(1):
            raise ValueError("S1 hidden alignment failure")
        wav=torch.as_tensor(load_audio(str(folder/"wav"/name),32000),dtype=torch.float32)[None]
        spec=spectrogram_torch(wav,2048,32000,640,2048,center=False)
        mel_spec=spectrogram_torch(wav,1280,32000,320,1280,center=False)
        mel=(spec_to_mel_torch(mel_spec,1280,100,32000,0,None)+12)/14*2-1
        # Only initial 1 second of reference audio; no future style leakage.
        refer=spec[...,:50]
        if min(mel.size(-1),y.size(1)*4)<110:
            raise ValueError("Too short for bridge evaluation")
        out.append(dict(codes=y[:,None].cpu(),hidden=hidden.cpu(),text=x.cpu(),
                        refer=refer.cpu(),mel=mel.cpu(),prompt_frames=100))
    return out


def to_device(example,device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in example.items()}


def condition(example,mode,global_mean):
    h=example["hidden"]
    if mode=="aligned":
        return example
    if mode=="prompt_mean":
        z=h[:,:min(25,h.size(1))].mean(1,keepdim=True)
    elif mode=="global_mean":
        z=global_mean.to(h.device)
    else:
        raise ValueError("Unknown bridge condition")
    return dict(example,hidden=z.expand_as(h).contiguous())


def evaluate(model,examples,device,mode="aligned",global_mean=None):
    scores={"hard":0.,"bridge":0.,"shuffled":0.}
    deltas=[]
    for i,ex in enumerate(examples):
        for seed in (8201+i,9201+i):
            example=condition(to_device(ex,device),mode,global_mean)
            row=paired_losses(model,example,seed)
            if mode=="aligned":
                shuffled=dict(example,hidden=example["hidden"].flip(1))
                row["shuffled"]=paired_losses(model,shuffled,seed)["bridge"]
            else:
                row["shuffled"]=row["bridge"]
            deltas.append(row["bridge"]-row["hard"])
            for k in scores: scores[k]+=row[k]/(2*len(examples))
    scores["paired_win_rate"]=sum(x<0 for x in deltas)/len(deltas)
    return scores


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--s2",type=Path,required=True)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--train-count",type=int,default=8)
    p.add_argument("--val-count",type=int,default=4)
    p.add_argument("--epochs",type=int,default=2)
    p.add_argument("--compare-initializations",action="store_true")
    p.add_argument("--compare-conditioning",action="store_true")
    p.add_argument("--test",type=Path)
    p.add_argument("--test-count",type=int,default=8)
    args=p.parse_args()
    torch.manual_seed(2027)
    device=torch.device(args.device)
    teacher,cfg,config=load_teacher(args.teacher,device)
    train,ti=dataset(args.train,cfg,config)
    val,vi=dataset(args.validation,cfg,config)
    if min(args.train_count,args.val_count,args.epochs)<1 or ti.keys() & vi.keys() or len(ti)<args.train_count or len(vi)<args.val_count:
        raise ValueError("Disjoint corpora with at least 8/4 utterances required")
    allow=not (config.get("text_features") or {}).get("multilingual_bert",False)
    a=extract(teacher,cfg,train,list(ti.values())[:args.train_count],args.train,device,allow)
    b=extract(teacher,cfg,val,list(vi.values())[:args.val_count],args.validation,device,allow)
    c=None
    if args.test:
        test,xi=dataset(args.test,cfg,config)
        if xi.keys() & (ti.keys() | vi.keys()) or len(xi)<args.test_count:
            raise ValueError("Independent test examples required")
        c=extract(teacher,cfg,test,list(xi.values())[:args.test_count],args.test,device,allow)
    global_mean=torch.stack([ex["hidden"][:,:min(25,ex["hidden"].size(1))].mean(1,keepdim=True)
                            for ex in a]).mean(0)
    del teacher
    torch.cuda.empty_cache()
    from module.models import SynthesizerTrnV3
    h=json.loads((ROOT/"GPT_SoVITS/configs/s2.json").read_text())
    s2=SynthesizerTrnV3(h["data"]["filter_length"]//2+1,h["train"]["segment_size"]//h["data"]["hop_length"],
                       n_speakers=h["data"]["n_speakers"],version="v5dev",**h["model"])
    state=torch.load(args.s2,map_location="cpu",weights_only=False)["weight"]
    result=s2.load_state_dict(state,strict=False)
    if any(k.startswith(("quantizer.","enc_p.","ref_enc.","bridge.","wns1.","cfm.")) for k in result.missing_keys):
        raise RuntimeError("Critical V5dev weights missing")
    del state
    args.output.mkdir(parents=True,exist_ok=False)
    arms=("zero_gate","zero_projection") if args.compare_initializations else ("zero_gate",)
    if args.compare_conditioning:
        arms=("aligned","prompt_mean","global_mean")
    results={}
    for arm in arms:
        torch.manual_seed(2027)
        mode=arm if args.compare_conditioning else "aligned"
        init="zero_projection" if args.compare_conditioning else arm
        model=FrozenV5Bridge(s2,SemanticResidualBridge(cfg.dim,768,128,initialization=init)).to(device).train()
        initial=evaluate(model,b,device,mode,global_mean)
        if abs(initial["hard"]-initial["bridge"])>1e-6:
            raise RuntimeError("Bridge zero parity failed")
        opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.adapter.parameters()),lr=3e-4,weight_decay=.01)
        best=initial["bridge"]
        rows=[]
        target=args.output/(arm+"-best.pt")
        for epoch in range(args.epochs):
            model.train()
            for i,ex in enumerate(a):
                # Matched noise across arms, distinct across examples and epochs.
                torch.manual_seed(2027+epoch*len(a)+i)
                opt.zero_grad(set_to_none=True)
                loss=model.loss(**condition(to_device(ex,device),mode,global_mean))
                if not torch.isfinite(loss): raise FloatingPointError("Bridge loss nonfinite")
                loss.backward()
                if any(p.grad is not None for p in model.s2.parameters()):
                    raise RuntimeError("S2 unexpectedly received weight gradients")
                torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),1.,error_if_nonfinite=True)
                opt.step()
            model.eval()
            metrics=evaluate(model,b,device,mode,global_mean)
            rows.append(metrics)
            print(json.dumps({"arm":arm,"epoch":epoch+1,**metrics}),flush=True)
            if metrics["bridge"]<best:
                best=metrics["bridge"]
                tmp=target.with_suffix(".tmp")
                torch.save({"format":"s1-next-bridge-pilot-v2","source":"original-s1-teacher",
                    "initialization":init,"condition":mode,"global_mean":global_mean,"adapter":model.adapter.state_dict(),"metrics":metrics},tmp)
                os.replace(tmp,target)
        results[arm]={"initial":initial,"epochs":rows,"best":best,
                      "gate":float(model.adapter.gate.detach()),"checkpoint_saved":target.exists()}
        if c:
            if target.exists():
                saved=torch.load(target,map_location="cpu",weights_only=True)
                model.adapter.load_state_dict(saved["adapter"],strict=True)
            else:
                # No validated improvement: evaluate the exact identity baseline.
                model.adapter=SemanticResidualBridge(cfg.dim,768,128,initialization=init).to(device)
            model.eval()
            results[arm]["test"]=evaluate(model,c,device,mode,global_mean)
        del model,opt
        torch.cuda.empty_cache()
    report={"status":"completed","arms":results,"train_count":len(a),"validation_count":len(b),"test_count":len(c) if c else 0}
    (args.output/"metrics.json").write_text(json.dumps(report,allow_nan=False,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
