"""Round 14: continue round 12 with affine RMS normalization homotopy.
The original mean-centering is gradually removed; original affine weights and
all trained SwiGLU weights are retained. No random reset or production changes.
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
from tools.train_s1_progressive_stack import input_state,state_cpu,logits
from tools.train_s1_fullwidth_ffn import loss_fn
from tools.audit_s1_next_transfer import scores
from tools.audit_s1_stack_kv import load_stack
from tools.train_s1_next_ddp import read_shards,save_atomic,batches,gradient_multiplier


class AffineRMS(nn.Module):
    def __init__(self,source):
        super().__init__()
        self.weight=nn.Parameter(source.weight.detach().clone())
        self.bias=nn.Parameter(source.bias.detach().clone())
        self.eps=source.eps
        self.normalized_shape=source.normalized_shape
        self.alpha=0.

    def forward(self,x):
        if self.alpha==0.:
            return F.layer_norm(x,self.normalized_shape,self.weight,self.bias,self.eps)
        if self.alpha==1.:
            z=x
        else:
            z=x-(1.-self.alpha)*x.mean(-1,keepdim=True)
        return z*torch.rsqrt(z.square().mean(-1,keepdim=True)+self.eps)*self.weight+self.bias


class HybridRMS(nn.Module):
    def __init__(self,teacher,mods):
        super().__init__()
        self.teacher=teacher
        self.ffns=nn.ModuleList([mods[i] for i in range(len(teacher.h.layers))])
        self.norms=nn.ModuleList([AffineRMS(n) for l in teacher.h.layers for n in (l.norm1,l.norm2)])
        self.alpha=0.

    def set_alpha(self,value):
        self.alpha=float(value)
        for m in self.norms:
            m.alpha=self.alpha

    def forward(self,e):
        teacher=self.teacher
        device=next(teacher.parameters()).device
        h,mask,_=input_state(teacher,e,device)
        for i,l in enumerate(teacher.h.layers):
            a=l.self_attn
            b,n,d=h.shape
            q,k,v=F.linear(h,a.in_proj_weight,a.in_proj_bias).view(b,n,3,a.num_heads,d//a.num_heads).permute(2,0,3,1,4).unbind(0)
            z=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
            h=self.norms[2*i](h+a.out_proj(z.transpose(1,2).reshape(b,n,d)))
            h=self.norms[2*i+1](h+self.ffns[i](h))
        if teacher.h.norm is not None:
            h=teacher.h.norm(h)
        return teacher.ar_predict_layer(h[:,e["x"].size(1)-1+e["prompt"]:])


@torch.no_grad()
def evaluate(model,rows,device,rank,world):
    sums=torch.zeros(10,device=device,dtype=torch.float64)
    for e in rows[rank::world]:
        z=model(e);t=e["teacher"].to(device);y=e["target"].to(device)
        assert z.shape==t.shape
        sums[:5]+=scores(z,t,y).to(device)
        sums[5:]+=scores(t,t,y).to(device)
    dist.all_reduce(sums)
    a,k,g,c,n,ta,tk,tg,tc,tn=sums.tolist()
    assert n==tn and n>0
    return {"ce":a/n,"teacher_ce":ta/n,"ce_ratio":a/ta,"kl":k/n,
            "top1_agreement":g/n,"target_accuracy":c/n,"teacher_accuracy":tc/n,"tokens":int(n)}


def passed(m):
    # Stricter than rounds 9-12: do not trade away the near-teacher result.
    return m["ce_ratio"]<=1.005 and m["kl"]<=.003 and m["top1_agreement"]>=.95


def snap(m):
    return {"ffns":state_cpu(m.ffns),"norms":state_cpu(m.norms)}


def restore(m,s,alpha):
    m.ffns.load_state_dict(s["ffns"],strict=True)
    m.norms.load_state_dict(s["norms"],strict=True)
    m.set_alpha(alpha)


def train(a):
    from tools.train_s1_next import load_teacher,teacher_logits
    rank=int(os.environ["RANK"]);local=int(os.environ["LOCAL_RANK"]);world=int(os.environ["WORLD_SIZE"])
    assert world==2 and torch.cuda.device_count()==2
    torch.cuda.set_device(local)
    device=torch.device("cuda",local)
    dist.init_process_group("nccl",timeout=timedelta(minutes=10))
    try:
        started=time.monotonic()
        torch.manual_seed(2034)
        teacher,cfg,_=load_teacher(a.teacher,device)
        teacher.eval().requires_grad_(False)
        mods=load_stack(teacher,a.stack,device)
        m=HybridRMS(teacher,mods).to(device).eval()
        m.ffns.requires_grad_(True)
        m.norms.requires_grad_(True)
        valid=read_shards(a.data,world,"validation")
        rows=read_shards(a.data,world,"train")
        random.Random(2034).shuffle(rows)
        rows=rows[:a.pairs]
        assert len(valid)==24 and len(rows)==a.pairs
        for e in valid[rank::world]:
            x,b,y=(e[k].to(device) for k in ("x","b","y"))
            t,_=teacher_logits(teacher,x,b,y)
            torch.testing.assert_close(t[:,e["prompt"]:],e["teacher"].to(device),rtol=3e-3,atol=3e-3)
            z,_=logits(teacher,e,mods,device)
            with torch.no_grad():
                torch.testing.assert_close(m(e),z[:,e["prompt"]:],rtol=3e-3,atol=3e-3)
        baseline=evaluate(m,valid,device,rank,world)
        assert passed(baseline)
        best=snap(m);current=baseline;alpha=0.
        initial={}
        for v in (.05,.1,.2,.35,.5,.75,1.):
            m.set_alpha(v)
            initial[str(v)]=evaluate(m,valid,device,rank,world)
        restore(m,best,alpha)
        history=[];tokens=0;steps=0;stop="full_rms"
        if rank==0:
            a.output.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        def persist():
            if rank==0:
                save_atomic({"format":"s1-affine-rms-v1","dim":cfg.dim,"depth":cfg.depth,
                    "width":m.ffns[0].up.out_features,"alpha":alpha,"state":best,
                    "validation":current},a.output/"norm-best.pt")
                r={"world_size":world,"precision":"fp32","training_pairs":len(rows),
                   "validation_pairs":len(valid),"baseline":baseline,"initial":initial,
                   "selected":current,"alpha":alpha,"full_rms":alpha==1.,
                   "norm_count":len(m.norms),"history":history,"tokens":tokens,"steps":steps,
                   "stop_reason":stop,"seconds":time.monotonic()-started}
                tmp=a.output/"metrics.tmp"
                tmp.write_text(json.dumps(r,indent=2,allow_nan=False))
                os.replace(tmp,a.output/"metrics.json")
        def expired():
            flag=torch.tensor(int(time.monotonic()-started>=a.seconds),device=device)
            dist.broadcast(flag,src=0)
            return bool(flag.item())
        persist()
        # Jump directly to the strongest passing initial value; then adaptive bisection.
        target=max([float(k) for k,v in initial.items() if passed(v)]+[.05])
        stage=0
        while alpha<1. and stage<10:
            if expired():
                stop="time_budget"
                break
            stage+=1
            restore(m,best,target)
            warm=evaluate(m,valid,device,rank,world)
            model=DDP(m,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True)
            params=list(m.ffns.parameters())+list(m.norms.parameters())
            opt=torch.optim.AdamW([{"params":m.ffns.parameters(),"lr":2e-6},
                                   {"params":m.norms.parameters(),"lr":1e-4}],weight_decay=0.,fused=True)
            streak=int(passed(warm));selected=None;selected_m=None
            epochs=[{"epoch":0,"validation":warm}]
            exhausted=False
            for epoch in range(1,a.epochs+1):
                order=list(range(len(rows)))
                random.Random(2034+stage*100+epoch).shuffle(order)
                totals=torch.zeros(2,device=device,dtype=torch.float64)
                for j,group in enumerate(batches(order,world,4)):
                    count=sum(rows[i]["target"].numel() for chunk in group for i in chunk)
                    opt.zero_grad(set_to_none=True)
                    chunk=group[rank]
                    for k,i in enumerate(chunk):
                        e=rows[i];t=e["teacher"].to(device);n=e["target"].numel()
                        sync=contextlib.nullcontext() if k==len(chunk)-1 else model.no_sync()
                        with sync:
                            loss=loss_fn(model(e),t)
                            (loss*n*gradient_multiplier(world,count)).backward()
                        totals[0]+=loss.detach().double()*n;totals[1]+=n
                    torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                    opt.step();tokens+=count;steps+=1
                    if (j+1)%8==0 and expired():
                        exhausted=True
                        break
                dist.all_reduce(totals)
                val=evaluate(m,valid,device,rank,world)
                streak=streak+1 if passed(val) else 0
                epochs.append({"epoch":epoch,"validation":val,"train_loss":float(totals[0]/totals[1]),
                               "partial_epoch":exhausted})
                if passed(val) and (selected_m is None or (val["kl"],-val["top1_agreement"])<(selected_m["kl"],-selected_m["top1_agreement"])):
                    selected=snap(m);selected_m=val
                if rank==0:
                    print(json.dumps({"event":"layer_epoch","stage":stage,"scale":target,
                        "epoch":epoch,"tokens":tokens,"validation":val,"gate_streak":streak}),flush=True)
                if streak>=2 or exhausted or expired():
                    break
            ok=streak>=2 and selected is not None
            if ok:
                best=selected;current=selected_m;alpha=target
            history.append({"stage":stage,"target_alpha":target,"accepted":ok,"epochs":epochs,
                            "selected_alpha":alpha,"selected":current})
            del model,opt,params,selected
            persist()
            dist.barrier()
            if exhausted:
                stop="time_budget"
                break
            if ok:
                if alpha==1.:
                    break
                target=min(1.,alpha+max(.1,(1.-alpha)*.4))
            else:
                if target-alpha<.025:
                    stop="minimum_step"
                    break
                target=(target+alpha)/2
        if alpha<1. and stop=="full_rms":
            stop="stage_budget"
        restore(m,best,alpha)
        final=evaluate(m,valid,device,rank,world)
        assert passed(final) and abs(final["ce"]-current["ce"])<1e-5
        current=final
        persist()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def selftest():
    torch.manual_seed(12)
    n=nn.LayerNorm(32)
    n.weight.data.uniform_(.5,1.5);n.bias.data.uniform_(-.3,.3)
    m=AffineRMS(n)
    x=torch.randn(2,9,32,requires_grad=True)+2
    torch.testing.assert_close(m(x),n(x))
    m.alpha=1.
    expected=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+n.eps)*n.weight+n.bias
    torch.testing.assert_close(m(x),expected)
    m.alpha=.25
    m(x).square().mean().backward()
    assert torch.isfinite(m.weight.grad).all()
    assert passed({"ce_ratio":1.001,"kl":.001,"top1_agreement":.97})
    assert not passed({"ce_ratio":1.001,"kl":.001,"top1_agreement":.94})
    print("LayerNorm identity, affine RMS endpoint, finite gradients and stricter gate passed")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    for name in ("teacher","data","stack","output"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--pairs",type=int,default=256)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--seconds",type=int,default=1000)
    a=p.parse_args()
    if a.selftest:
        selftest()
    else:
        train(a)
