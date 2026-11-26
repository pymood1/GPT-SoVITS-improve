"""Round 12: full-width weight-preserving 24-layer SwiGLU conversion.
Keep every original FFN neuron; anneal the gate from near-ReLU to standard SiLU.
Frozen attention, norms, embeddings and output head. All GPU work is bounded.
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
from tools.train_s1_progressive_ffn import passed, SwiFFN
from tools.train_s1_progressive_stack import evaluate, logits, rollout, state_cpu
from tools.train_s1_joint_ffn import Hybrid, distillation, quality_key, clone_replacements
from tools.train_s1_next_ddp import read_shards, save_atomic, batches, gradient_multiplier


class FullFFN(nn.Module):
    def __init__(self,d,w,scale=128.):
        super().__init__()
        self.gate_proj=nn.Linear(d,w)
        self.up=nn.Linear(d,w)
        self.down_proj=nn.Linear(w,d)
        self.scale=float(scale)

    @torch.no_grad()
    def initialize(self,layer):
        # No pruning, no random trainable weights remain.
        self.gate_proj.load_state_dict(layer.linear1.state_dict(),strict=True)
        self.down_proj.load_state_dict(layer.linear2.state_dict(),strict=True)
        self.up.weight.zero_()
        self.up.bias.fill_(1.)

    def gate(self,x):
        return self.gate_proj(x)*self.scale

    def down(self,x):
        return self.down_proj(x/self.scale)

    def forward(self,x):
        return self.down(F.silu(self.gate(x))*self.up(x))

    @torch.no_grad()
    def folded(self):
        # Export ordinary SwiGLU, absorbing scale into the original linear maps.
        return {"gate.weight":self.gate_proj.weight.detach().cpu()*self.scale,
                "gate.bias":self.gate_proj.bias.detach().cpu()*self.scale,
                "up.weight":self.up.weight.detach().cpu().clone(),
                "up.bias":self.up.bias.detach().cpu().clone(),
                "down.weight":self.down_proj.weight.detach().cpu()/self.scale,
                "down.bias":self.down_proj.bias.detach().cpu().clone()}


def full_modules(teacher,device,scale,states=None):
    mods={}
    for i,l in enumerate(teacher.h.layers):
        m=FullFFN(l.linear1.in_features,l.linear1.out_features,scale).to(device)
        if states is None:
            m.initialize(l)
        else:
            m.load_state_dict(states[i],strict=True)
        mods[i]=m
    return mods


def snapshot(mods):
    return {i:state_cpu(m) for i,m in mods.items()}


def loss_fn(z,t):
    z,t=z.float(),t.float()
    # Retain KL, and explicitly preserve top-1 ordering on close teacher logits.
    v,idx=t.topk(2,dim=-1)
    top=z.gather(-1,idx[...,:1]).squeeze(-1)
    other=z.scatter(-1,idx[...,:1],-torch.inf).amax(-1)
    margin=(v[...,0]-v[...,1]).clamp(max=.15)
    return distillation(z,t)+.1*F.relu(margin-(top-other)).square().mean()


def train(args):
    from tools.train_s1_next import load_teacher,teacher_logits
    rank=int(os.environ["RANK"]);local=int(os.environ["LOCAL_RANK"]);world=int(os.environ["WORLD_SIZE"])
    assert world==2 and torch.cuda.device_count()==2
    torch.cuda.set_device(local)
    device=torch.device("cuda",local)
    dist.init_process_group("nccl",timeout=timedelta(minutes=10))
    try:
        started=time.monotonic()
        torch.manual_seed(2031)
        teacher,cfg,_=load_teacher(args.teacher,device)
        teacher.eval().requires_grad_(False)
        saved=torch.load(args.first,map_location="cpu",weights_only=True)
        assert saved["format"]=="s1-progressive-stack-v1" and saved["dim"]==cfg.dim
        assert saved["width"]==cfg.ff_dim and saved["depth"]==cfg.depth
        assert sorted(saved["replacements"])==[0,1,2,3,4,10,11] and passed(saved["validation"])
        rows=read_shards(args.data,world,"train")
        random.Random(2031).shuffle(rows)
        rows=rows[:args.pairs]
        valid=read_shards(args.data,world,"validation")
        assert len(rows)==args.pairs and len(valid)==24
        for e in valid[rank::world]:
            x,b,y=(e[k].to(device) for k in ("x","b","y"))
            t,_=teacher_logits(teacher,x,b,y)
            z,_=logits(teacher,e,{},device)
            torch.testing.assert_close(t,z,rtol=3e-3,atol=3e-3)
            torch.testing.assert_close(t[:,e["prompt"]:],e["teacher"].to(device),rtol=3e-3,atol=3e-3)
        prior=clone_replacements(saved["replacements"],cfg,device)
        baseline=evaluate(teacher,valid,prior,device,rank,world)
        del prior
        mods=full_modules(teacher,device,128.)
        widths={m.up.out_features for m in mods.values()}
        assert len(widths)==1
        width=widths.pop()
        initial={}
        for scale in (128.,32.,8.,2.,1.):
            for m in mods.values():
                m.scale=scale
            initial[str(scale)]=evaluate(teacher,valid,mods,device,rank,world)
            if rank==0:
                print(json.dumps({"event":"scale_audit","scale":scale,"validation":initial[str(scale)]}),flush=True)
        # Select the smoothest passing untrained initialization (if any).
        eligible=[s for s in (128.,32.,8.,2.,1.) if passed(initial[str(s)])]
        best_scale=min(eligible) if eligible else 128.
        best_m=initial[str(best_scale)]
        best=snapshot(mods)
        gate_met=passed(best_m)
        original_params=sum(p.numel() for p in teacher.parameters())
        old_ffn=sum(p.numel() for l in teacher.h.layers for m in (l.linear1,l.linear2) for p in m.parameters())
        new_ffn=sum(p.numel() for m in mods.values() for p in m.parameters())
        history=[];tokens=0;steps_total=0;stop="schedule_completed"
        if rank==0:
            args.output.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        def persist():
            if rank==0:
                save_atomic({"format":"s1-fullwidth-ffn-v1","dim":cfg.dim,"width":width,
                             "depth":cfg.depth,"scale":best_scale,"replacements":best,
                             "validation":best_m,"gate_met":gate_met},args.output/"stack-best.pt")
                report={"world_size":world,"precision":"fp32","training_pairs":len(rows),
                        "validation_pairs":len(valid),"prior":baseline,"initial_scales":initial,
                        "selected":best_m,"selected_scale":best_scale,"gate_met":gate_met,
                        "accepted_layers":list(range(cfg.depth)) if gate_met else [],
                        "original_ffn_width":width,"compact_ffn_width":cfg.ff_dim,
                        "original_model_parameters":original_params,
                        "full_model_parameters":original_params-old_ffn+new_ffn,
                        "history":history,"tokens":tokens,"steps":steps_total,
                        "stop_reason":stop,"seconds":time.monotonic()-started}
                tmp=args.output/"metrics.tmp"
                tmp.write_text(json.dumps(report,indent=2,allow_nan=False))
                os.replace(tmp,args.output/"metrics.json")
        def expired():
            flag=torch.tensor(int(time.monotonic()-started>=args.seconds),device=device)
            dist.broadcast(flag,src=0)
            return bool(flag.item())
        persist()
        del mods
        # Calibrate even if standard SiLU already passes; otherwise progressively smooth.
        schedule=[best_scale]+[s for s in (32.,8.,2.,1.) if s<best_scale]
        for stage,scale in enumerate(schedule):
            if expired():
                stop="time_budget"
                break
            mods=full_modules(teacher,device,scale,best)
            warm=evaluate(teacher,valid,mods,device,rank,world)
            stage_best=None;stage_m=None;stage_epoch=None
            if passed(warm):
                stage_best=snapshot(mods);stage_m=warm;stage_epoch=0
            hybrid=Hybrid(teacher,mods).eval()
            model=DDP(hybrid,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True)
            params=list(hybrid.ffns.parameters())
            opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.,fused=True)
            epochs=[{"epoch":0,"validation":warm}]
            exhausted=False
            streak=0
            for epoch in range(1,args.epochs+1):
                order=list(range(len(rows)))
                random.Random(2031+stage*100+epoch).shuffle(order)
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
                            loss=loss_fn(z,t)
                            (loss*n*gradient_multiplier(world,count)).backward()
                        totals[0]+=loss.detach().double()*n;totals[1]+=n
                    torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                    opt.step()
                    tokens+=count;steps_total+=1
                    if (step+1)%8==0 and expired():
                        exhausted=True
                        break
                dist.all_reduce(totals)
                m=evaluate(teacher,valid,mods,device,rank,world)
                streak=streak+1 if passed(m) else 0
                epochs.append({"epoch":epoch,"validation":m,"train_loss":float(totals[0]/totals[1]),
                               "partial_epoch":exhausted})
                if passed(m) and (stage_m is None or quality_key(m)<quality_key(stage_m)):
                    stage_best=snapshot(mods);stage_m=m;stage_epoch=epoch
                if rank==0:
                    print(json.dumps({"event":"layer_epoch","stage":stage,"scale":scale,"epoch":epoch,
                                      "tokens":tokens,"validation":m,"gate_streak":streak}),flush=True)
                if streak>=2 or exhausted or expired():
                    break
            # A smoother stage can be accepted only with a passing trained check;
            # otherwise preserve the last passing checkpoint, including pretraining.
            ok=stage_best is not None and (streak>=2 or (passed(warm) and streak>=1))
            if ok and (scale<best_scale or not gate_met or quality_key(stage_m)<quality_key(best_m)):
                best=stage_best;best_m=stage_m;best_scale=scale;gate_met=True
            history.append({"stage":stage,"scale":scale,"accepted":ok,"selected_epoch":stage_epoch if ok else None,
                            "epochs":epochs,"selected_scale":best_scale,"selected":best_m})
            del model,hybrid,opt,params,mods,stage_best
            persist()
            dist.barrier()
            if exhausted:
                stop="time_budget"
                break
        final=full_modules(teacher,device,best_scale,best)
        measured=evaluate(teacher,valid,final,device,rank,world)
        assert abs(measured["ce"]-best_m["ce"])<1e-5
        assert passed(measured)==gate_met
        # Check actual ordinary SwiGLU export, not just the scaled training parametrization.
        folded={}
        for i,m in final.items():
            f=SwiFFN(cfg.dim,width).to(device)
            f.load_state_dict(m.folded(),strict=True)
            folded[i]=f
        exported=evaluate(teacher,valid,folded,device,rank,world)
        assert abs(exported["ce"]-measured["ce"])<1e-4
        assert abs(exported["kl"]-measured["kl"])<1e-4
        persist()
        if rank==0:
            save_atomic({"format":"s1-progressive-stack-v1","dim":cfg.dim,"width":width,
                         "depth":cfg.depth,"replacements":{i:m.folded() for i,m in final.items()},
                         "validation":exported},args.output/"folded-stack.pt")
            path=args.output/"metrics.json"
            report=json.loads(path.read_text())
            report["exported"]=exported
            report["export_gate_met"]=passed(exported)
            report["all_ffn_replaced"]=gate_met and passed(exported)
            path.write_text(json.dumps(report,indent=2,allow_nan=False))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def selftest():
    from types import SimpleNamespace
    torch.manual_seed(31)
    l=SimpleNamespace(linear1=nn.Linear(16,64),linear2=nn.Linear(64,16))
    m=FullFFN(16,64,128.)
    m.initialize(l)
    x=torch.randn(256,16)
    t=l.linear2(F.relu(l.linear1(x))).detach()
    assert torch.equal(m.gate_proj.weight,l.linear1.weight)
    assert torch.equal(m.down_proj.weight,l.linear2.weight)
    assert (m.up.weight==0).all() and (m.up.bias==1).all()
    assert F.mse_loss(m(x),t)<1e-5
    f=SwiFFN(16,64)
    f.load_state_dict(m.folded(),strict=True)
    torch.testing.assert_close(f(x),m(x),rtol=1e-5,atol=1e-6)
    z=torch.randn(1,9,11,requires_grad=True)
    y=torch.randn_like(z)
    loss_fn(z,y).backward()
    assert torch.isfinite(z.grad).all()
    print("Full-width transfer, folded export and ranking loss checks passed")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    for name in ("teacher","data","first","output"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--pairs",type=int,default=384)
    p.add_argument("--micro",type=int,default=4)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--lr",type=float,default=1e-5)
    p.add_argument("--seconds",type=int,default=1200)
    args=p.parse_args()
    if args.selftest:
        selftest()
    else:
        train(args)
