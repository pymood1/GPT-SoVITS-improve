"""Round 11: jointly distill all converted FFNs against the original output distribution.
Start from round 10; add locally initialized groups, accept only cumulative gated checkpoints.
No production changes. All validation-guided decisions use the same 24 replay prompts.
"""
import argparse
import contextlib
import json
import os
import random
import time
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tools.train_s1_progressive_ffn import SwiFFN, passed
from tools.train_s1_progressive_stack import (advance, pre_ffn, original_ffn, input_state,
    evaluate, state_cpu, score, rollout, logits)
from tools.train_s1_next_ddp import read_shards, save_atomic, batches, gradient_multiplier

# One-based order selected from round 10 rejected candidates' best agreement.
ORDER=[12,15,19,20,6,13,17,22,21,24,9,16,23,8,18,14,10,7]


class Hybrid(nn.Module):
    def __init__(self,teacher,replacements):
        super().__init__()
        self.teacher=teacher
        self.ffns=nn.ModuleDict({str(i):m for i,m in replacements.items()})

    def forward(self,e):
        teacher=self.teacher
        device=next(teacher.parameters()).device
        h,mask,_=input_state(teacher,e,device)
        for i,layer in enumerate(teacher.h.layers):
            ffn=self.ffns[str(i)] if str(i) in self.ffns else None
            h=advance(h,layer,mask,ffn)
        if teacher.h.norm is not None:
            h=teacher.h.norm(h)
        return teacher.ar_predict_layer(h[:,e["x"].size(1)-1+e["prompt"]:])


def distillation(z,t):
    # Match the sampled policy's original distribution; also match its softer tail.
    z,t=z.float(),t.float()
    def kl(temp):
        return F.kl_div(F.log_softmax(z/temp,-1),F.softmax(t/temp,-1),
                        reduction="sum")*temp**2
    return (kl(1.)+.25*kl(2.))/t.shape[-2]


def clone_replacements(states,cfg,device):
    out={}
    for i,state in states.items():
        m=SwiFFN(cfg.dim,cfg.ff_dim).to(device)
        m.load_state_dict(state,strict=True)
        out[int(i)]=m
    return out


def snapshot(mods):
    return {i:state_cpu(m) for i,m in mods.items()}


def quality_key(m):
    # Prefer actual agreement, break ties by teacher-distribution divergence.
    return (-m["top1_agreement"],m["kl"],m["ce"])


def improved(m,base):
    return passed(m) and m["top1_agreement"]>=base["top1_agreement"] and m["kl"]<base["kl"]


@torch.no_grad()
def local_features(teacher,rows,indices,device):
    # Original activations only, cached per group. No validation samples.
    hs={i:[] for i in indices};fs={i:[] for i in indices}
    last=max(indices)
    for e in rows:
        h,mask,_=input_state(teacher,e,device)
        for i,layer in enumerate(teacher.h.layers):
            if i in hs:
                z=pre_ffn(h,layer,mask)
                f=original_ffn(layer,z)
                hs[i].append(z.squeeze(0));fs[i].append(f.squeeze(0))
                h=layer.norm2(z+f,None)
            else:
                h=advance(h,layer,mask)
            if i==last:
                break
    return {i:(torch.cat(hs[i]),torch.cat(fs[i])) for i in indices}


def initialize_group(teacher,mods,indices,rows,cfg,device,rank,world):
    features=local_features(teacher,rows[rank::world],indices,device)
    for i in indices:
        layer=teacher.h.layers[i]
        ffn=SwiFFN(cfg.dim,cfg.ff_dim).to(device)
        ffn.warm_start(layer)
        model=DDP(ffn,device_ids=[device.index],broadcast_buffers=False)
        h,f=features.pop(i)
        size=torch.tensor(len(h),device=device)
        dist.all_reduce(size,op=dist.ReduceOp.MIN)
        steps=int(size.item())//1024
        assert steps>0
        opt=torch.optim.AdamW(ffn.parameters(),lr=3e-4,weight_decay=.001,fused=True)
        for _ in range(2):
            order=torch.randperm(len(h),device=device)
            for j in range(steps):
                idx=order[j*1024:(j+1)*1024]
                a,t=h[idx],f[idx]
                pred=model(a)
                loss=(pred-t).square().mean()/t.square().mean().clamp_min(1e-5)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ffn.parameters(),1.,error_if_nonfinite=True)
                opt.step()
        mods[i]=ffn.eval()
        del model,opt,h,f
    return mods


def train(args):
    from tools.train_s1_next import load_teacher,teacher_logits
    rank=int(os.environ["RANK"]);local=int(os.environ["LOCAL_RANK"]);world=int(os.environ["WORLD_SIZE"])
    assert world==2 and torch.cuda.device_count()==2
    torch.cuda.set_device(local)
    device=torch.device("cuda",local)
    dist.init_process_group("nccl",timeout=timedelta(minutes=10))
    try:
        started=time.monotonic()
        torch.manual_seed(2030)
        teacher,cfg,_=load_teacher(args.teacher,device)
        teacher.eval().requires_grad_(False)
        saved=torch.load(args.first,map_location="cpu",weights_only=True)
        assert saved["format"]=="s1-progressive-stack-v1" and saved["dim"]==cfg.dim
        assert saved["width"]==cfg.ff_dim and saved["depth"]==cfg.depth
        assert sorted(saved["replacements"])==[0,1,2,3,4,10] and passed(saved["validation"])
        accepted=saved["replacements"]
        rows=read_shards(args.data,world,"train")
        random.Random(2030).shuffle(rows)
        rows=rows[:args.pairs]
        valid=read_shards(args.data,world,"validation")
        assert len(rows)==args.pairs and len(valid)==24
        for e in valid[rank::world]:
            x,b,y=(e[k].to(device) for k in ("x","b","y"))
            t,_=teacher_logits(teacher,x,b,y)
            z,_=logits(teacher,e,{},device)
            torch.testing.assert_close(t,z,rtol=3e-3,atol=3e-3)
            torch.testing.assert_close(t[:,e["prompt"]:],e["teacher"].to(device),rtol=3e-3,atol=3e-3)
        current=evaluate(teacher,valid,clone_replacements(accepted,cfg,device),device,rank,world)
        assert passed(current)
        initial=current
        history=[]; tokens=0; steps_total=0
        stop="all_groups_attempted"
        if rank==0:
            args.output.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        def persist():
            if rank==0:
                save_atomic({"format":"s1-progressive-stack-v1","dim":cfg.dim,"width":cfg.ff_dim,
                             "depth":cfg.depth,"replacements":accepted,"validation":current},
                            args.output/"stack-best.pt")
                report={"world_size":world,"precision":"fp32","training_pairs":len(rows),
                        "validation_pairs":len(valid),"initial":initial,"selected":current,
                        "accepted_layers":sorted(accepted),"history":history,"tokens":tokens,
                        "steps":steps_total,"stop_reason":stop,"seconds":time.monotonic()-started}
                tmp=args.output/"metrics.tmp"
                tmp.write_text(json.dumps(report,indent=2,allow_nan=False))
                os.replace(tmp,args.output/"metrics.json")
        def expired():
            flag=torch.tensor(int(time.monotonic()-started>=args.seconds),device=device)
            dist.broadcast(flag,src=0)
            return bool(flag.item())
        persist()
        groups=[[]]+[[v-1 for v in ORDER[i:i+3]] for i in range(0,len(ORDER),3)]
        stage=0
        while groups and stage<10:
            if expired():
                stop="time_budget"
                break
            indices=groups.pop(0)
            stage+=1
            mods=clone_replacements(accepted,cfg,device)
            if indices:
                mods=initialize_group(teacher,mods,indices,rows[:128],cfg,device,rank,world)
            warm=evaluate(teacher,valid,mods,device,rank,world)
            epochs=[{"epoch":0,"validation":warm}]
            best=None;best_m=None;best_epoch=None
            hybrid=Hybrid(teacher,mods)
            hybrid.eval()  # frozen teacher dropout must remain disabled
            model=DDP(hybrid,device_ids=[local],broadcast_buffers=False,
                      gradient_as_bucket_view=True)
            params=list(hybrid.ffns.parameters())
            assert all(p.requires_grad for p in params)
            assert not any(p.requires_grad for p in teacher.parameters())
            opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=.001,fused=True)
            exhausted=False
            streak=0
            for epoch in range(1,args.epochs+1):
                order=list(range(len(rows)))
                random.Random(2030+stage*100+epoch).shuffle(order)
                totals=torch.zeros(2,device=device,dtype=torch.float64)
                for step,group in enumerate(batches(order,world,args.micro)):
                    count=sum(rows[i]["target"].numel() for chunk in group for i in chunk)
                    opt.zero_grad(set_to_none=True)
                    chunk=group[rank]
                    for j,i in enumerate(chunk):
                        e=rows[i];t=e["teacher"].to(device)
                        n=e["target"].numel()
                        sync=contextlib.nullcontext() if j==len(chunk)-1 else model.no_sync()
                        with sync:
                            z=model(e)
                            assert z.shape==t.shape
                            loss=distillation(z,t)
                            (loss*n*gradient_multiplier(world,count)).backward()
                        totals[0]+=loss.detach().double()*n
                        totals[1]+=n
                    torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                    opt.step()
                    tokens+=count;steps_total+=1
                    if (step+1)%8==0 and expired():
                        exhausted=True
                        break
                dist.all_reduce(totals)
                m=evaluate(teacher,valid,mods,device,rank,world)
                eligible=passed(m) if indices else improved(m,current)
                streak=streak+1 if eligible else 0
                epochs.append({"epoch":epoch,"validation":m,"train_loss":float(totals[0]/totals[1]),
                               "eligible":eligible,"partial_epoch":exhausted})
                if eligible and (best_m is None or quality_key(m)<quality_key(best_m)):
                    best=snapshot(mods);best_m=m;best_epoch=epoch
                if rank==0:
                    print(json.dumps({"event":"layer_epoch","stage":stage,"group":indices,
                                      "epoch":epoch,"tokens":tokens,"validation":m,"gate_streak":streak},allow_nan=False),flush=True)
                if streak>=2 or exhausted or expired():
                    break
            # Require two consecutive passing trained checks; rollback the entire group.
            ok=streak>=2 and best is not None
            if ok:
                accepted=best;current=best_m
            elif len(indices)>1 and not exhausted:
                # Split rejected groups to locate sensitive layers; never silently relax the gate.
                groups.append(indices[:1]);groups.append(indices[1:])
            history.append({"stage":stage,"group":indices,"accepted":ok,"selected_epoch":best_epoch if ok else None,
                            "selected":current,"epochs":epochs})
            del model,hybrid,opt,params,mods
            persist()
            if rank==0:
                print(json.dumps({"event":"layer_result","stage":stage,"group":indices,"accepted":ok,
                                  "accepted_count":len(accepted),"validation":current},allow_nan=False),flush=True)
            dist.barrier()
            if exhausted:
                stop="time_budget"
                break
        if groups and stop!="time_budget":
            stop="stage_budget"
        final_mods=clone_replacements(accepted,cfg,device)
        final=evaluate(teacher,valid,final_mods,device,rank,world)
        assert passed(final) and abs(final["ce"]-current["ce"])<1e-5
        current=final
        persist()
        prior=clone_replacements(saved["replacements"],cfg,device)
        trial=[]
        for e in valid[:4][rank::world]:
            seed=40100+e["index"]
            t,ts=rollout(teacher,e,{},device,seed)
            a,ats=rollout(teacher,e,prior,device,seed)
            b,bs=rollout(teacher,e,final_mods,device,seed)
            def summary(v,seconds):
                n=min(t.size(1),v.size(1))
                return {"tokens":v.size(1),"positional_matches":int((t[:,:n]==v[:,:n]).sum()),
                        "compared":n,"seconds":seconds}
            trial.append({"index":e["index"],"teacher_tokens":t.size(1),"teacher_seconds":ts,
                          "prior":summary(a,ats),"candidate":summary(b,bs)})
        gathered=[None]*world
        dist.all_gather_object(gathered,trial)
        if rank==0:
            path=args.output/"metrics.json"
            report=json.loads(path.read_text())
            report["rollouts"]=[r for shard in gathered for r in shard]
            report["all_ffn_replaced"]=len(accepted)==cfg.depth
            report["seconds"]=time.monotonic()-started
            path.write_text(json.dumps(report,indent=2,allow_nan=False))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def selftest():
    torch.manual_seed(5)
    t=torch.randn(1,17,11)
    z=t.clone().requires_grad_(True)
    assert abs(float(distillation(z,t)))<1e-6
    loss=distillation(z+.3*torch.randn_like(z),t)
    loss.backward()
    assert loss>0 and torch.isfinite(z.grad).all()
    assert len(ORDER)==18 and set(ORDER)|{1,2,3,4,5,11}==set(range(1,25))
    a={"ce":2.85,"ce_ratio":1.002,"kl":.008,"top1_agreement":.902}
    assert improved({**a,"kl":.003,"top1_agreement":.95},a)
    assert not improved({**a,"kl":.003,"top1_agreement":.89},a)
    print("Joint distillation loss, stage order and acceptance tests passed")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    for name in ("teacher","data","first","output"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--pairs",type=int,default=256)
    p.add_argument("--micro",type=int,default=4)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--lr",type=float,default=3e-5)
    p.add_argument("--seconds",type=int,default=1500)
    args=p.parse_args()
    if args.selftest:
        selftest()
    else:
        train(args)
