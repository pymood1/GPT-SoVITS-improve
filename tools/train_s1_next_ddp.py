"""Two-GPU prompt distillation: sharded teacher generation, synchronous DDP training.

One student parameter set, replicated on both devices. Global token weighted
gradient accumulation, no repeated padding samples, rank-zero checkpoints.
"""
import argparse
import contextlib
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from AR.models.s1_next import NextConfig,S1Next,kd_loss,sample_next
from tools.train_s1_next import teacher_logits,teacher_generate
from tools.train_s1_next_prompt import make_pairs,load_rows,target_slice,evaluate
from tools.evaluate_s1_next_waveforms import features,normalized


def batches(order,world,micro):
    """Equal collective count; final remainder is processed, never duplicated."""
    if world<1 or micro<1 or len(order)<world:
        raise ValueError("Insufficient examples")
    # Only an indivisible final sample (at most world-1) is rotated each epoch.
    usable=len(order)//world*world
    for start in range(0,usable,world*micro):
        group=order[start:min(usable,start+world*micro)]
        yield [group[r::world] for r in range(world)]


def gradient_multiplier(world,global_tokens):
    if world<1 or global_tokens<1:
        raise ValueError("Invalid token budget")
    return world/global_tokens


@torch.no_grad()
def cached_teacher_generate(teacher,x,b,prompt,*,cap,seed,verify=False):
    """Same logits and sampling policy as full-forward reference, with KV cache."""
    import torch.nn.functional as F
    t=teacher.ar_text_position(teacher.ar_text_embedding(x)+teacher.bert_proj(b.transpose(1,2)))
    p=teacher.ar_audio_position(teacher.ar_audio_embedding(prompt))
    n,m=x.size(1),prompt.size(1)
    xt=torch.cat((t,p),1)
    mask=torch.cat((F.pad(torch.zeros(n,n,dtype=torch.bool,device=x.device),(0,m),value=True),
                    F.pad(torch.ones(m,m,dtype=torch.bool,device=x.device).triu(1),(n,0),value=False)),0)
    mask=mask[None,None].expand(1,teacher.num_head,-1,-1)
    h,k,v=teacher.t2s_transformer.process_prompt(xt,mask,None)
    g=torch.Generator(device=x.device).manual_seed(seed)
    y=prompt
    for i in range(cap):
        z=teacher.ar_predict_layer(h[:,-1])
        if verify and i<8:
            ref,_=teacher_logits(teacher,x,b,y)
            torch.testing.assert_close(z,ref[:,-1],rtol=2e-3,atol=2e-3)
        token=sample_next(z,y,eos=teacher.EOS,allow_eos=i>=10,top_k=15,
                          repetition_penalty=1.35,generator=g)
        if token.item()==teacher.EOS:
            break
        y=torch.cat((y,token),1)
        # The position module refreshes its sinusoid buffer if needed.
        pos=teacher.ar_audio_position(teacher.ar_audio_embedding(y))[:,-1:]
        h,k,v=teacher.t2s_transformer.decode_next_token(pos,k,v)
    return y[:,m:]


def save_atomic(payload,path):
    tmp=path.with_suffix(".tmp")
    torch.save(payload,tmp)
    os.replace(tmp,path)


def prepare_worker(args):
    """Each worker creates its own exact FP32 targets; save sharded cache."""
    device=torch.device("cuda:0")
    from TTS_infer_pack.TTS import TTS,TTS_Config
    train_rows,val_rows=load_rows(args.train),load_rows(args.validation)
    if {normalized(r["text"]) for r in train_rows}&{normalized(r["text"]) for r in val_rows}:
        raise ValueError("Train/validation overlap")
    args.output.mkdir(parents=True,exist_ok=False)
    stats={}
    cached=True
    checked=False
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ddp-pseudo-") as tmp:
        conf=TTS_Config({"custom":dict(TTS_Config.default_configs["v5dev"],device="cuda:0",is_half=False,
                         t2s_weights_path=str(args.teacher.resolve()),vits_weights_path=str(args.s2.resolve()))})
        conf.configs_path=str(Path(tmp)/"tts.yaml")
        engine=TTS(conf)
        teacher=engine.t2s_model.model.eval()
        for label,rows,count,seed,pair_seed in (("train",train_rows,args.train_pairs,33000,2027),
                                                ("validation",val_rows,args.val_pairs,40100,2028)):
            examples=[]
            attempts=0
            for i,(ref,target) in enumerate(make_pairs(rows,count,pair_seed)):
                if i%args.world!=args.rank:
                    continue
                attempts+=1
                with torch.no_grad():
                    engine.set_ref_audio(ref["path"])
                    px,pb=features(engine,ref["text"])
                    tx,tb=features(engine,target["text"])
                    x,b=torch.cat((px,tx),1),torch.cat((pb,tb),2)
                    prompt=engine.prompt_cache["prompt_semantic"].reshape(1,-1).long()
                    if cached:
                        try:
                            y=cached_teacher_generate(teacher,x,b,prompt,cap=384,seed=seed+i,verify=not checked)
                            checked=True
                        except (AssertionError,AttributeError,TypeError,RuntimeError) as exc:
                            if isinstance(exc,torch.cuda.OutOfMemoryError):
                                raise
                            cached=False
                            print(json.dumps({"event":"cache_fallback","reason":type(exc).__name__}),flush=True)
                    if not cached:
                        y=teacher_generate(teacher,x,b,prompt,cap=384,top_k=15,penalty=1.35,seed=seed+i)
                    if 12<=y.size(1)<384:
                        context=torch.cat((prompt,y),1)
                        logits,targets=teacher_logits(teacher,x,b,context)
                        z,t=target_slice(logits,targets,prompt.size(1))
                        examples.append(dict(index=i,x=x.cpu(),b=b.cpu(),y=context.cpu(),prompt=int(prompt.size(1)),
                                             teacher=z.float().cpu(),target=t.cpu(),target_count=int(y.size(1))))
                print(json.dumps({"event":"pseudo_pair","split":label,"accepted":len(examples),
                                  "attempts":attempts}),flush=True)
                # Persist completed shards; no giant per-step rewrites.
                if attempts%64==0:
                    save_atomic(examples,args.output/(label+".pt"))
            save_atomic(examples,args.output/(label+".pt"))
            if len(examples)<max(4,int(.75*attempts)):
                raise RuntimeError("Insufficient valid pseudo trajectories")
            stats[label]={"accepted":len(examples),"attempts":attempts}
    stats.update(cache_verified=checked,cache_enabled=cached,seconds=time.monotonic()-started)
    (args.output/"metrics.json").write_text(json.dumps(stats,allow_nan=False))


def read_shards(root,world,split):
    rows=[]
    for r in range(world):
        rows.extend(torch.load(root/f"pseudo{r}"/(split+".pt"),map_location="cpu",weights_only=True))
    rows.sort(key=lambda e:e["index"])
    if len({e["index"] for e in rows})!=len(rows):
        raise ValueError("Duplicate shard indices")
    return rows


def evaluate_distributed(model,rows,device,rank,world):
    local=rows[rank::world]
    if not local:
        raise ValueError("Empty validation shard")
    metrics=evaluate(model,local,device,True)
    keys=sorted(metrics)
    packed=torch.tensor([metrics[k]*len(local) for k in keys]+[len(local)],device=device,dtype=torch.float64)
    dist.all_reduce(packed)
    return {k:float(packed[i]/packed[-1]) for i,k in enumerate(keys)}


def train_worker(args):
    rank=int(os.environ["RANK"])
    local=int(os.environ["LOCAL_RANK"])
    world=int(os.environ["WORLD_SIZE"])
    if world!=2 or torch.cuda.device_count()!=2:
        raise RuntimeError("Expected two synchronous GPU replicas")
    torch.cuda.set_device(local)
    device=torch.device("cuda",local)
    from datetime import timedelta
    dist.init_process_group("nccl",timeout=timedelta(minutes=10))
    try:
        torch.manual_seed(2027)
        teacher_cfg=torch.load(args.teacher,map_location="cpu",weights_only=False)["config"]
        payload=torch.load(args.student,map_location="cpu",weights_only=False)
        cfg=NextConfig.from_teacher(teacher_cfg)
        if payload["format"]!="s1-next-v1" or payload["teacher_config"]!=teacher_cfg or payload["config"]!=asdict(cfg):
            raise ValueError("Checkpoint contract mismatch")
        student=S1Next(cfg)
        student.load_state_dict(payload["state_dict"],strict=True)
        del payload
        student=student.to(device)
        model=DDP(student,device_ids=[local],output_device=local,
                  gradient_as_bucket_view=True,broadcast_buffers=False)
        # FP32 master weights, gradients, optimizer and KD, no unverified AMP.
        opt=torch.optim.AdamW(student.parameters(),lr=args.lr,weight_decay=.01,fused=True)
        train=read_shards(args.data,world,"train")
        valid=read_shards(args.data,world,"validation")
        if rank==0:
            args.output.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        history=[]
        count=0
        steps=0
        started=time.monotonic()
        best=float("inf")
        def record(epoch,metrics,seconds,loss=None):
            nonlocal best
            row={"epoch":epoch,"validation":metrics,"tokens":count,"steps":steps,
                 "seconds":seconds,"tokens_per_second":count/max(.001,time.monotonic()-started)}
            if loss is not None:
                row["train_loss"]=loss
            history.append(row)
            score=metrics["kd"]+2*metrics["short_rate"]+2*metrics["cap_rate"]+abs(metrics["mean_length_ratio"]-1)
            if rank==0:
                if score<best:
                    best=score
                    save_atomic({"format":"s1-next-v1","config":asdict(cfg),"state_dict":student.state_dict(),
                                 "teacher_config":teacher_cfg,"epoch":epoch,"tokens":count,
                                 "validation":metrics},args.output/"s1-next-best.pt")
                print(json.dumps(row,allow_nan=False),flush=True)
                report={"world_size":world,"precision":"fp32","train_pairs":len(train),
                        "val_pairs":len(valid),"history":history,
                        "seconds":time.monotonic()-started,"selected_score":best}
                tmp=args.output/"metrics.tmp"
                tmp.write_text(json.dumps(report,allow_nan=False))
                os.replace(tmp,args.output/"metrics.json")
            dist.barrier()
        record(0,evaluate_distributed(student,valid,device,rank,world),0.)
        for epoch in range(1,args.epochs+1):
            t=time.monotonic()
            model.train()
            order=list(range(len(train)))
            random.Random(2027+epoch).shuffle(order)
            local_loss=0.
            local_tokens=0
            for group in batches(order,world,args.micro):
                indices=group[rank]
                total=sum(train[i]["target"].numel() for chunk in group for i in chunk)
                opt.zero_grad(set_to_none=True)
                for j,i in enumerate(indices):
                    e=train[i]
                    x,b,y=(e[k].to(device) for k in ("x","b","y"))
                    target=e["target"].to(device)
                    teacher=e["teacher"].to(device)
                    sync=contextlib.nullcontext() if j==len(indices)-1 else model.no_sync()
                    with sync:
                        logits,_=model(x,b,y)
                        loss,_,_=kd_loss(logits[:,e["prompt"]:],teacher,target,temperature=2.,alpha=.7)
                        n=target.numel()
                        (loss*n*gradient_multiplier(world,total)).backward()
                    local_loss+=float(loss.detach())*n
                    local_tokens+=n
                torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                count+=total
                steps+=1
                if rank==0 and steps%25==0:
                    print(json.dumps({"event":"train_step","epoch":epoch,"step":steps,"tokens":count}),flush=True)
            packed=torch.tensor([local_loss,local_tokens],device=device,dtype=torch.float64)
            dist.all_reduce(packed)
            metrics=evaluate_distributed(student,valid,device,rank,world)
            record(epoch,metrics,time.monotonic()-t,float(packed[0]/packed[1]))
    finally:
        dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["prepare","train"])
    for name in ("teacher","student","s2","train","validation","data","output"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--rank",type=int,default=0)
    p.add_argument("--world",type=int,default=2)
    p.add_argument("--train-pairs",type=int,default=1024)
    p.add_argument("--val-pairs",type=int,default=24)
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--micro",type=int,default=4)
    p.add_argument("--lr",type=float,default=5e-5)
    a=p.parse_args()
    (prepare_worker if a.mode=="prepare" else train_worker)(a)

if __name__=="__main__":
    main()
