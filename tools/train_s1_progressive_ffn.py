"""Round 9: preserve teacher except first FFN, fit a compact SwiGLU with 2-GPU DDP.
Local hidden regression, then held-out whole-model logits. Research checkpoint only.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from AR.models.s1_next import prefix_mask
from tools.audit_s1_next_transfer import variant, scores
from tools.train_s1_next_ddp import read_shards, save_atomic


class SwiFFN(nn.Module):
    def __init__(self,d,w):
        super().__init__()
        self.gate=nn.Linear(d,w)
        self.up=nn.Linear(d,w)
        self.down=nn.Linear(w,d)

    def forward(self,x):
        return self.down(F.silu(self.gate(x))*self.up(x))

    @torch.no_grad()
    def warm_start(self,layer,scale=4.):
        # Keep high-weight-product neurons; SiLU(scale*z)/scale approximates ReLU(z).
        a,b=layer.linear1,layer.linear2
        importance=a.weight.norm(dim=1)*b.weight.norm(dim=0)
        idx=importance.topk(self.gate.out_features).indices
        self.gate.weight.copy_(a.weight[idx]*scale)
        self.gate.bias.copy_(a.bias[idx]*scale)
        self.up.weight.zero_()
        self.up.bias.fill_(1.)
        self.down.weight.copy_(b.weight[:,idx]/scale)
        self.down.bias.copy_(b.bias)


@torch.no_grad()
def first_features(teacher,e,device):
    x,b,y=(e[k].to(device) for k in ("x","b","y"))
    n=x.size(1)
    xy,mask,_=teacher.make_input_data(x,torch.tensor([n],device=device),y,torch.tensor([y.size(1)],device=device),b)
    l=teacher.h.layers[0]
    h=l.norm1(xy+l._sa_block(xy,mask,None),None)
    f=l.linear2(l.activation(l.linear1(h)))
    return h.squeeze(0),f.squeeze(0)


@torch.no_grad()
def evaluate(teacher,ffn,rows,device,rank,world):
    from tools.train_s1_next import teacher_logits
    sums=torch.zeros(10,dtype=torch.float64,device=device)
    for e in rows[rank::world]:
        x,b,y=(e[k].to(device) for k in ("x","b","y"))
        p=e["prompt"]
        t,tar=teacher_logits(teacher,x,b,y)
        target=tar[:,p:]
        assert torch.equal(target,e["target"].to(device))
        torch.testing.assert_close(t[:,p:],e["teacher"].to(device),rtol=3e-3,atol=3e-3)
        z,_=variant(teacher,x,b,y,"fresh_ffn",SimpleNamespace(blocks=[ffn]),only=0)
        sums[:5]+=scores(z[:,p:],t[:,p:],target).to(device)
        sums[5:]+=scores(t[:,p:],t[:,p:],target).to(device)
    dist.all_reduce(sums)
    a,k,g,c,n,ta,tk,tg,tc,tn=sums.tolist()
    assert n==tn and n>0
    return {"ce":a/n,"teacher_ce":ta/n,"ce_ratio":a/ta,"kl":k/n,
            "top1_agreement":g/n,"target_accuracy":c/n,"teacher_accuracy":tc/n,"tokens":int(n)}


def passed(m):
    return m["ce_ratio"]<=1.01 and m["kl"]<=.02 and m["top1_agreement"]>=.90


def train(args):
    from tools.train_s1_next import load_teacher
    rank=int(os.environ["RANK"])
    local=int(os.environ["LOCAL_RANK"])
    world=int(os.environ["WORLD_SIZE"])
    assert world==2 and torch.cuda.device_count()==2
    torch.cuda.set_device(local)
    device=torch.device("cuda",local)
    from datetime import timedelta
    dist.init_process_group("nccl",timeout=timedelta(minutes=5))
    try:
        started=time.monotonic()
        torch.manual_seed(2028)
        teacher,cfg,_=load_teacher(args.teacher,device)
        teacher.eval().requires_grad_(False)
        rows=read_shards(args.data,world,"train")
        random.Random(2028).shuffle(rows)
        rows=rows[:args.pairs]
        valid=read_shards(args.data,world,"validation")
        assert len(rows)==args.pairs and len(valid)==24
        # Source round 6 verifies disjoint utterance splits; pair indices are split-local.
        layer=teacher.h.layers[0]
        ffn=SwiFFN(cfg.dim,cfg.ff_dim).to(device)
        initial_random=evaluate(teacher,ffn,valid,device,rank,world)
        ffn.warm_start(layer)
        initial_warm=evaluate(teacher,ffn,valid,device,rank,world)
        model=DDP(ffn,device_ids=[local],broadcast_buffers=False)
        if rank==0:
            args.output.mkdir(parents=True,exist_ok=False)
            print(json.dumps({"event":"initial","random":initial_random,"warm":initial_warm}),flush=True)
        hs,fs=[],[]
        for e in rows[rank::world]:
            h,f=first_features(teacher,e,device)
            hs.append(h)
            fs.append(f)
        h=torch.cat(hs); f=torch.cat(fs)
        del hs,fs,rows
        num=torch.tensor(len(h),device=device)
        dist.all_reduce(num,op=dist.ReduceOp.MIN)
        steps=int(num.item())//args.batch
        assert steps>0
        opt=torch.optim.AdamW(ffn.parameters(),lr=args.lr,weight_decay=.001,fused=True)
        norm=layer.norm2
        history=[]
        best=float("inf")
        streak=0
        tokens=0
        stop="epoch_budget"
        def record(epoch,m,loss=None):
            nonlocal best,streak
            streak=streak+1 if passed(m) else 0
            row={"epoch":epoch,"tokens":tokens,"validation":m,"train_loss":loss,
                 "lr":opt.param_groups[0]["lr"],"gate_streak":streak}
            history.append(row)
            # Prefer teacher distribution, then CE; no selection on held-out audio quality.
            score=m["kl"]+.1*max(0.,m["ce"]-m["teacher_ce"])
            if score<best:
                best=score
                if rank==0:
                    save_atomic({"format":"s1-progressive-ffn-v1","layer":0,"dim":cfg.dim,
                                 "width":cfg.ff_dim,"state_dict":ffn.state_dict(),"validation":m,"epoch":epoch},
                                args.output/"ffn-best.pt")
            if rank==0:
                print(json.dumps(row,allow_nan=False),flush=True)
        record(0,initial_warm)
        for epoch in range(1,args.epochs+1):
            order=torch.randperm(len(h),device=device)
            ls=torch.zeros((),device=device)
            model.train()
            for i in range(steps):
                j=order[i*args.batch:(i+1)*args.batch]
                a,t=h[j],f[j]
                pred=model(a)
                # Scale the raw FFN error by teacher energy and match actual post-norm output.
                with torch.no_grad():
                    post=norm(a+t,None)
                loss=(pred-t).square().mean()/t.square().mean().clamp_min(1e-5)+F.mse_loss(norm(a+pred,None),post)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ffn.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                ls+=loss.detach()
                tokens+=args.batch*world
                if rank==0 and (i+1)%25==0:
                    print(json.dumps({"event":"train_step","epoch":epoch,"step":(epoch-1)*steps+i+1,"tokens":tokens}),flush=True)
            dist.all_reduce(ls)
            model.eval()
            record(epoch,evaluate(teacher,ffn,valid,device,rank,world),float(ls/(steps*world)))
            flag=torch.tensor(int(time.monotonic()-started>=args.seconds),device=device)
            dist.broadcast(flag,src=0)
            if streak>=2:
                stop="gate_met"
                break
            if flag.item():
                stop="time_budget"
                break
        if rank==0:
            report={"world_size":world,"precision":"fp32","layer":0,"training_pairs":args.pairs,
                    "validation_pairs":len(valid),"initial_random":initial_random,
                    "initial_warm":initial_warm,"history":history,"stop_reason":stop,
                    "selected":torch.load(args.output/"ffn-best.pt",map_location="cpu",weights_only=True)["validation"],
                    "seconds":time.monotonic()-started,"gate_met":streak>=2}
            (args.output/"metrics.json").write_text(json.dumps(report,indent=2,allow_nan=False))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def selftest():
    torch.manual_seed(4)
    l=SimpleNamespace(linear1=nn.Linear(16,64),linear2=nn.Linear(64,16))
    x=torch.randn(256,16)
    f=SwiFFN(16,64)
    t=l.linear2(F.relu(l.linear1(x))).detach()
    f.warm_start(l,scale=20.)
    assert F.mse_loss(f(x),t).item()<.001
    f(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in f.parameters())
    assert passed({"ce_ratio":1.001,"kl":.01,"top1_agreement":.95})
    assert not passed({"ce_ratio":1.02,"kl":.01,"top1_agreement":.95})
    print("Progressive FFN initialization and gradient tests passed")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    for name in ("teacher","data","output"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--pairs",type=int,default=512)
    p.add_argument("--batch",type=int,default=1024)
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--seconds",type=int,default=600)
    args=p.parse_args()
    if args.selftest:
        selftest()
    else:
        train(args)
