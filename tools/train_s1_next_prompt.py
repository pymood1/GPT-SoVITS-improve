"""Prompt-conditioned distillation: match the independent-reference inference distribution."""
import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"GPT_SoVITS")]
os.environ.setdefault("version","v2")
os.environ.setdefault("hz","25hz")
os.environ.setdefault("language","ko_KR")
import torch
from torch.nn import functional as F
from AR.models.s1_next import NextConfig,S1Next,kd_loss
from tools.train_s1_next import teacher_generate,teacher_logits
from tools.evaluate_s1_next_waveforms import features,normalized


def make_pairs(rows,count,seed):
    if len(rows)<4 or count<1:
        raise ValueError("Need independent reference and target candidates")
    shuffled=list(rows)
    random.Random(seed).shuffle(shuffled)
    split=len(shuffled)//2
    refs,targets=shuffled[:split],shuffled[split:]
    pairs=[]
    for i in range(count):
        ref=refs[i%len(refs)]
        target=targets[(i+(i//len(refs)))%len(targets)]
        if normalized(ref["text"])==normalized(target["text"]):
            raise ValueError("Reference/target transcript overlap")
        pairs.append((ref,target))
    if len({(r["path"],t["path"]) for r,t in pairs})!=len(pairs):
        raise ValueError("Duplicate prompt pair")
    return pairs


def target_slice(logits,targets,prompt_length):
    if not 0<prompt_length<targets.size(1)-1:
        raise ValueError("Invalid target boundary")
    return logits[:,prompt_length:],targets[:,prompt_length:]


def load_rows(path):
    rows=json.loads(path.read_text())
    for row in rows:
        if not Path(row["path"]).is_file() or not normalized(row["text"]):
            raise ValueError("Invalid corpus row")
    return rows


@torch.no_grad()
def generate_pairs(engine,teacher,pairs,seed,cap,folder):
    examples=[]
    attempts=0
    for i,(ref,target) in enumerate(pairs):
        attempts+=1
        engine.set_ref_audio(ref["path"])
        px,pb=features(engine,ref["text"])
        tx,tb=features(engine,target["text"])
        x=torch.cat((px,tx),1)
        b=torch.cat((pb,tb),2)
        prompt=engine.prompt_cache["prompt_semantic"].reshape(1,-1).long()
        y=teacher_generate(teacher,x,b,prompt,cap=cap,top_k=15,penalty=1.35,seed=seed+i)
        if y.size(1)<12 or y.size(1)>=cap:
            continue
        context=torch.cat((prompt,y),1)
        logits,targets=teacher_logits(teacher,x,b,context)
        z,t=target_slice(logits,targets,prompt.size(1))
        examples.append(dict(x=x.cpu(),b=b.cpu(),y=context.cpu(),prompt=int(prompt.size(1)),
                             teacher=z.float().cpu(),target=t.cpu(),target_count=int(y.size(1))))
        print(json.dumps({"event":"pseudo_pair","accepted":len(examples),"attempts":attempts}),flush=True)
    folder.mkdir(parents=True,exist_ok=True)
    torch.save(examples,folder/"examples.pt")
    if len(examples)<max(4,int(.75*len(pairs))):
        raise RuntimeError("Too few complete teacher trajectories")
    return examples,attempts


def loss_for(model,e,device):
    x,b,y=(e[k].to(device) for k in ("x","b","y"))
    logits,_=model(x,b,y)
    z=logits[:,e["prompt"]:]
    t=e["target"].to(device)
    teacher=e["teacher"].to(device)
    loss,ce,kl=kd_loss(z,teacher,t,temperature=2.,alpha=.7)
    with torch.no_grad():
        eos=model.cfg.codes
        eos_prob=z[:,-1].float().softmax(-1)[0,eos]
        start_eos=z[:,0].float().softmax(-1)[0,eos]
        # CE is measured against pseudo labels, not ground-truth audio.
    return loss,float(ce),float(kl),float(start_eos),float(eos_prob)


@torch.no_grad()
def evaluate(model,examples,device,rollouts=False):
    model.eval()
    totals={"ce":0.,"kd":0.,"start_eos":0.,"end_eos":0.}
    ratios=[]
    caps=0
    for i,e in enumerate(examples):
        _,ce,kl,start,end=loss_for(model,e,device)
        for key,value in zip(totals,(ce,kl,start,end)):
            totals[key]+=value/len(examples)
        if rollouts:
            x,b,y=(e[k].to(device) for k in ("x","b","y"))
            g=torch.Generator(device=device).manual_seed(40100+i)
            cap=384
            ids,_=model.generate(x,b,y[:,:e["prompt"]],max_new=cap,min_new=10,
                                 top_k=15,repetition_penalty=1.35,generator=g)
            ratios.append(ids.size(1)/e["target_count"])
            caps+=ids.size(1)==cap
    if ratios:
        totals.update(mean_length_ratio=sum(ratios)/len(ratios),short_rate=sum(r<.5 for r in ratios)/len(ratios),
                      cap_rate=caps/len(ratios))
    return totals


def main():
    p=argparse.ArgumentParser()
    for name in ("teacher","student","s2","train","validation","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--train-pairs",type=int,default=96)
    p.add_argument("--val-pairs",type=int,default=12)
    p.add_argument("--epochs",type=int,default=12)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    device=torch.device("cuda:0")
    torch.manual_seed(2027)
    train_rows=load_rows(args.train)
    val_rows=load_rows(args.validation)
    if {normalized(r["text"]) for r in train_rows}&{normalized(r["text"]) for r in val_rows}:
        raise ValueError("Train validation transcript overlap")
    from TTS_infer_pack.TTS import TTS,TTS_Config
    teacher_cfg=torch.load(args.teacher,map_location="cpu",weights_only=False)["config"]
    payload=torch.load(args.student,map_location="cpu",weights_only=False)
    cfg=NextConfig.from_teacher(teacher_cfg)
    if payload["format"]!="s1-next-v1" or payload["teacher_config"]!=teacher_cfg or payload["config"]!=asdict(cfg):
        raise ValueError("Checkpoint contract mismatch")
    student=S1Next(cfg)
    student.load_state_dict(payload["state_dict"],strict=True)
    del payload
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix="prompt-distill-") as tmp:
        conf=TTS_Config({"custom":dict(TTS_Config.default_configs["v5dev"],device="cuda:0",is_half=False,
                        t2s_weights_path=str(args.teacher.resolve()),vits_weights_path=str(args.s2.resolve()))})
        conf.configs_path=str(Path(tmp)/"tts.yaml")
        engine=TTS(conf)
        teacher=engine.t2s_model.model.eval()
        train,nt=generate_pairs(engine,teacher,make_pairs(train_rows,args.train_pairs,2027),33000,384,
                                args.output/"train-pseudo")
        valid,nv=generate_pairs(engine,teacher,make_pairs(val_rows,args.val_pairs,2028),40100,384,
                                args.output/"val-pseudo")
        del teacher,engine
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    student=student.to(device)
    opt=torch.optim.AdamW(student.parameters(),lr=5e-5,weight_decay=.01,fused=True)
    history=[]
    initial=evaluate(student,valid,device,True)
    best=float("inf")
    count=0
    checkpoint=args.output/"s1-next-best.pt"
    def save(epoch,metrics):
        tmp=checkpoint.with_suffix(".tmp")
        torch.save({"format":"s1-next-v1","config":asdict(cfg),"state_dict":student.state_dict(),
                    "teacher_config":teacher_cfg,"epoch":epoch,"tokens":count,"validation":metrics},tmp)
        os.replace(tmp,checkpoint)
    def rank(m):
        return m["kd"]+2*m["short_rate"]+2*m["cap_rate"]+abs(m["mean_length_ratio"]-1)
    best=rank(initial)
    save(0,initial)
    history.append({"epoch":0,"validation":initial})
    for epoch in range(1,args.epochs+1):
        student.train()
        order=list(range(len(train)))
        random.Random(2027+epoch).shuffle(order)
        pending=0
        opt.zero_grad(set_to_none=True)
        for i in order:
            e=train[i]
            loss,*_=loss_for(student,e,device)
            n=e["target"].numel()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite prompt KD")
            (loss*n).backward()
            pending+=n
            count+=n
            if pending>=1024 or i==order[-1]:
                for p in student.parameters():
                    if p.grad is not None:
                        p.grad.div_(pending)
                torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                opt.zero_grad(set_to_none=True)
                pending=0
        metrics=evaluate(student,valid,device,True)
        history.append({"epoch":epoch,"validation":metrics,"tokens":count})
        if rank(metrics)<best:
            best=rank(metrics)
            save(epoch,metrics)
        print(json.dumps(history[-1],allow_nan=False),flush=True)
        (args.output/"metrics.json").write_text(json.dumps({"history":history,"train_pairs":len(train),
              "val_pairs":len(valid),"train_attempts":nt,"val_attempts":nv,"seconds":time.monotonic()-started},allow_nan=False))
    # Pseudo data are not needed as downloadable artifacts.
    for split in ("train-pseudo","val-pseudo"):
        (args.output/split/"examples.pt").unlink(missing_ok=True)

if __name__=="__main__":
    main()
