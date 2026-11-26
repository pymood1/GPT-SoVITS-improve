"""Offline S1 Next distillation on explicitly separated preprocessed corpora.

Run only with trusted local teacher checkpoints and private data.
Not a replacement for the production trainer. No S2 or dataset changes.
"""
import argparse
import hashlib
import math
import time
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"GPT_SoVITS")]
os.environ.setdefault("version","v2")
os.environ.setdefault("hz","25hz")

import torch
from torch.nn import functional as F
from AR.models.s1_next import NextConfig,S1Next,kd_loss


def teacher_logits(teacher,text,bert,audio):
    with torch.no_grad():
        x_len=torch.tensor([text.shape[1]],device=text.device)
        y_len=torch.tensor([audio.shape[1]],device=audio.device)
        xy,mask,target=teacher.make_input_data(text,x_len,audio,y_len,bert)
        h,_=teacher.h((xy,None),mask=mask)
        logits=teacher.ar_predict_layer(h[:,text.shape[1]-1:])
        expected=F.pad(audio,(0,1),value=teacher.EOS)
        if not torch.equal(expected,target):
            raise ValueError("Teacher target/EOS alignment mismatch")
        return logits,target


def load_teacher(path,device):
    from AR.models.t2s_lightning_module import Text2SemanticLightningModule
    payload=torch.load(path,map_location="cpu",weights_only=False)
    config=payload["config"]
    teacher=Text2SemanticLightningModule(config,"****",is_train=False)
    teacher.load_state_dict(payload["weight"],strict=True)
    teacher=teacher.model.to(device).eval().requires_grad_(False)
    return teacher,NextConfig.from_teacher(config),config


def dataset(path,cfg,teacher_config):
    from AR.data.dataset import Text2SemanticDataset
    data=Text2SemanticDataset(phoneme_path=str(path/"2-name2text.txt"),
        semantic_path=str(path/"6-name2semantic.tsv"),
        max_sec=teacher_config["data"]["max_sec"],pad_val=cfg.codes)
    # The legacy dataset repeats small corpora. Use each source ID only once.
    unique={}
    for i,name in enumerate(data.item_names):
        unique.setdefault(str(name),i)
    if not unique:
        raise ValueError("Empty corpus")
    return data,unique


def prepare(sample,cfg,device,allow_missing_bert=False):
    x=torch.as_tensor(sample["phoneme_ids"],dtype=torch.long,device=device)[None]
    y=torch.as_tensor(sample["semantic_ids"],dtype=torch.long,device=device)[None]
    b=sample["bert_feature"]
    if b is None:
        if not allow_missing_bert:
            raise ValueError("Missing BERT features; use explicit override only for compatible teacher")
        b=torch.zeros(cfg.bert_dim,x.size(1))
    b=torch.as_tensor(b,dtype=torch.float32,device=device)[None]
    if b.shape!=(1,cfg.bert_dim,x.size(1)) or not torch.isfinite(b).all():
        raise ValueError("Invalid BERT features")
    if x.numel()==0 or y.numel()==0 or x.min()<0 or x.max()>=cfg.phonemes or y.min()<0 or y.max()>=cfg.codes:
        raise ValueError("Invalid token IDs")
    return x,b,y


def checkpoint_digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


class TeacherCache:
    """Exact FP32 CPU logits, content-addressed by teacher and actual input tensors."""
    def __init__(self,path,teacher_digest):
        self.path=Path(path) if path else None
        self.identity=teacher_digest
        self.hits=self.misses=0
        if self.path:
            self.path.mkdir(parents=True,exist_ok=True)

    def get(self,teacher,x,b,y):
        if self.path is None:
            self.misses+=1
            return teacher_logits(teacher,x,b,y)
        h=hashlib.sha256(("s1-next-kd-v1:"+self.identity).encode())
        for v in (x,b,y):
            c=v.detach().contiguous().cpu()
            h.update(str((tuple(c.shape),str(c.dtype))).encode())
            h.update(c.numpy().tobytes())
        p=self.path/(h.hexdigest()+".pt")
        expected=F.pad(y,(0,1),value=teacher.EOS)
        if p.exists():
            cached=torch.load(p,map_location="cpu",weights_only=True)
            if cached.shape != (*expected.shape,teacher.EOS+1) or cached.dtype!=torch.float32 or not torch.isfinite(cached).all():
                raise ValueError("Invalid teacher cache entry")
            self.hits+=1
            return cached.to(x.device),expected
        logits,target=teacher_logits(teacher,x,b,y)
        self.misses+=1
        tmp=p.with_suffix(".%s.tmp"%os.getpid())
        try:
            torch.save(logits.detach().float().cpu(),tmp)
            os.replace(tmp,p)
        finally:
            tmp.unlink(missing_ok=True)
        return logits,target


def sample_loss(student,teacher,sample,device,args,cache=None):
    x,b,y=prepare(sample,student.cfg,device,args.allow_missing_bert)
    t,target=cache.get(teacher,x,b,y) if cache else teacher_logits(teacher,x,b,y)
    logits,_=student(x,b,y)
    loss,ce,kl=kd_loss(logits,t,target,temperature=args.temperature,alpha=args.alpha)
    with torch.no_grad():
        teacher_ce=F.cross_entropy(t.float().flatten(0,1),target.flatten())
        correct=(logits.argmax(-1)==target).sum()
    return loss,{"ce":float(ce),"kd":float(kl),"teacher_ce":float(teacher_ce),
                 "correct":int(correct),"tokens":target.numel()}


@torch.no_grad()
def evaluate(student,teacher,data,indices,device,args,cache=None):
    student.eval()
    result={"ce":0.,"kd":0.,"teacher_ce":0.,"correct":0,"tokens":0}
    for i in indices:
        _,m=sample_loss(student,teacher,data[i],device,args,cache)
        for key in ("ce","kd","teacher_ce"):
            result[key]+=m[key]*m["tokens"]
        for key in ("correct","tokens"):
            result[key]+=m[key]
    for key in ("ce","kd","teacher_ce"):
        result[key]/=result["tokens"]
    result["accuracy"]=result["correct"]/result["tokens"]
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--max-steps",type=int,default=1000)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--alpha",type=float,default=.5)
    p.add_argument("--temperature",type=float,default=2.)
    p.add_argument("--seed",type=int,default=2027)
    p.add_argument("--allow-missing-bert",action="store_true")
    p.add_argument("--random-init",action="store_true")
    p.add_argument("--teacher-cache",type=Path)
    p.add_argument("--tokens-per-update",type=int,default=2048)
    p.add_argument("--max-tokens",type=int,default=1000000)
    p.add_argument("--warmup-steps",type=int,default=50)
    args=p.parse_args()
    if min(args.epochs,args.max_steps,args.tokens_per_update,args.max_tokens)<1 or args.warmup_steps<0 or args.lr<=0 or not 0<=args.alpha<=1 or args.temperature<=0:
        p.error("Invalid training settings")
    if args.train.resolve()==args.validation.resolve():
        p.error("Use separate preprocessed training and validation corpora")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device=torch.device(args.device)
    teacher,cfg,teacher_config=load_teacher(args.teacher,device)
    cache=TeacherCache(args.teacher_cache,checkpoint_digest(args.teacher) if args.teacher_cache else '')
    train,train_ids=dataset(args.train,cfg,teacher_config)
    valid,valid_ids=dataset(args.validation,cfg,teacher_config)
    if train_ids.keys() & valid_ids.keys():
        raise ValueError("Training/validation IDs overlap")
    student=S1Next(cfg).to(device)
    if not args.random_init:
        copied=student.initialize_from_teacher(teacher)
        print("Copied compatible tensors:",copied,flush=True)
    optimizer=torch.optim.AdamW(student.parameters(),lr=args.lr,weight_decay=.01,fused=(device.type=='cuda'))
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint=args.output/"s1-next-best.pt"
    if checkpoint.exists():
        raise FileExistsError("Use a fresh output directory; no silent overwrite")
    val_indices=list(valid_ids.values())
    started=time.monotonic()
    initial=evaluate(student,teacher,valid,val_indices,device,args,cache)
    print(json.dumps({"event":"initial","validation":initial}),flush=True)
    best=initial["ce"]
    step=0
    total_tokens=0
    # Baseline is saved for reproducibility; a later checkpoint replaces it only on held-out improvement.
    def save(epoch,metrics):
        tmp=checkpoint.with_suffix(".tmp")
        torch.save({"format":"s1-next-v1","config":asdict(cfg),"state_dict":student.state_dict(),
                    "step":step,"epoch":epoch,"tokens":total_tokens,"validation":metrics,
                    "teacher_config":teacher_config},tmp)
        os.replace(tmp,checkpoint)
    save(0,initial)
    history=[{"epoch":0,"step":0,"tokens":0,"validation":initial}]
    def update(tokens):
        nonlocal step
        # Each microbatch loss was multiplied by its token count; normalize once.
        for p in student.parameters():
            if p.grad is not None:
                p.grad.div_(tokens)
        torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
        progress=min(1.,step/max(1,args.max_steps))
        warm=min(1.,(step+1)/max(1,args.warmup_steps))
        optimizer.param_groups[0]["lr"]=args.lr*warm*(.1+.9*.5*(1+math.cos(math.pi*progress)))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step+=1

    for epoch in range(1,args.epochs+1):
        student.train()
        indices=list(train_ids.values())
        random.Random(args.seed+epoch).shuffle(indices)
        pending=0
        optimizer.zero_grad(set_to_none=True)
        for i in indices:
            if step>=args.max_steps or total_tokens>=args.max_tokens:
                break
            loss,m=sample_loss(student,teacher,train[i],device,args,cache)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            (loss*m["tokens"]).backward()
            pending+=m["tokens"]
            total_tokens+=m["tokens"]
            if pending>=args.tokens_per_update or total_tokens>=args.max_tokens:
                update(pending)
                pending=0
                if step%25==0:
                    print(json.dumps({"step":step,"tokens":total_tokens,"ce":m["ce"],"kd":m["kd"]}),flush=True)
        if pending and step<args.max_steps:
            update(pending)
        metrics=evaluate(student,teacher,valid,val_indices,device,args,cache)
        print(json.dumps({"epoch":epoch,"step":step,"tokens":total_tokens,"validation":metrics,
                          "cache_hits":cache.hits,"cache_misses":cache.misses,
                          "seconds":time.monotonic()-started}),flush=True)
        history.append({"epoch":epoch,"step":step,"tokens":total_tokens,"validation":metrics})
        (args.output/"metrics.json").write_text(json.dumps({"status":"completed_epoch","history":history,
                    "cache_hits":cache.hits,"cache_misses":cache.misses,"seconds":time.monotonic()-started},allow_nan=False,indent=2))
        if metrics["ce"]<best:
            best=metrics["ce"]
            save(epoch,metrics)
        if step>=args.max_steps or total_tokens>=args.max_tokens:
            break



if __name__=="__main__":
    main()
