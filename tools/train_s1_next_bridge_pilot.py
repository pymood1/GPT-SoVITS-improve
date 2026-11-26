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


def evaluate(model,examples,device):
    scores={"hard":0.,"bridge":0.}
    for i,ex in enumerate(examples):
        for seed in (8201+i,9201+i):
            row=paired_losses(model,to_device(ex,device),seed)
            for k in scores: scores[k]+=row[k]/(2*len(examples))
    return scores


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--s2",type=Path,required=True)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args()
    torch.manual_seed(2027)
    device=torch.device(args.device)
    teacher,cfg,config=load_teacher(args.teacher,device)
    train,ti=dataset(args.train,cfg,config)
    val,vi=dataset(args.validation,cfg,config)
    if ti.keys() & vi.keys() or len(ti)<8 or len(vi)<4:
        raise ValueError("Disjoint corpora with at least 8/4 utterances required")
    allow=not (config.get("text_features") or {}).get("multilingual_bert",False)
    a=extract(teacher,cfg,train,list(ti.values())[:8],args.train,device,allow)
    b=extract(teacher,cfg,val,list(vi.values())[:4],args.validation,device,allow)
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
    model=FrozenV5Bridge(s2,SemanticResidualBridge(cfg.dim,768,128)).to(device).train()
    initial=evaluate(model,b,device)
    if abs(initial["hard"]-initial["bridge"])>1e-6:
        raise RuntimeError("Bridge zero parity failed")
    opt=torch.optim.AdamW(model.adapter.parameters(),lr=3e-4,weight_decay=.01)
    args.output.mkdir(parents=True,exist_ok=False)
    best=initial["bridge"]
    rows=[]
    for epoch in range(2):
        for ex in a:
            opt.zero_grad(set_to_none=True)
            loss=model.loss(**to_device(ex,device))
            if not torch.isfinite(loss): raise FloatingPointError("Bridge loss nonfinite")
            loss.backward()
            if any(p.grad is not None for p in model.s2.parameters()):
                raise RuntimeError("S2 unexpectedly received weight gradients")
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),1.,error_if_nonfinite=True)
            opt.step()
        metrics=evaluate(model,b,device)
        rows.append(metrics)
        if metrics["bridge"]<best:
            best=metrics["bridge"]
            tmp=args.output/"bridge.tmp"
            torch.save({"format":"s1-next-bridge-pilot-v1","source":"original-s1-teacher",
                "adapter":model.adapter.state_dict(),"metrics":metrics},tmp)
            os.replace(tmp,args.output/"bridge-best.pt")
    report={"status":"completed","initial":initial,"epochs":rows,"best":best,
            "train_count":len(a),"validation_count":len(b),"gate":float(model.adapter.gate.detach()),
            "checkpoint_saved":(args.output/"bridge-best.pt").exists()}
    (args.output/"metrics.json").write_text(json.dumps(report,allow_nan=False,indent=2))
    print(json.dumps(report),flush=True)

if __name__=="__main__":
    main()
