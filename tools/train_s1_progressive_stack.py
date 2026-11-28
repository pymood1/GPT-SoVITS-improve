"""Round 10: cumulative, gated FFN migration. Frozen original S1; two-GPU DDP.
Cache original and accepted-prefix features, fit only one new FFN at a time.
Checkpoint contains replacements only; no change to production inference.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from AR.models.s1_next import prefix_mask, sample_next
from tools.audit_s1_next_transfer import block, scores
from tools.train_s1_next_ddp import read_shards, save_atomic
from tools.train_s1_progressive_ffn import SwiFFN, passed


def input_state(teacher,e,device):
    x,b,y=(e[k].to(device) for k in ("x","b","y"))
    n=x.size(1)
    h,_,target=teacher.make_input_data(x,torch.tensor([n],device=device),
                y,torch.tensor([y.size(1)],device=device),b)
    return h,prefix_mask(n,y.size(1),device),target


def advance(h,layer,mask,ffn=None):
    return block(h,layer,mask,"identity",ffn)


def pre_ffn(h,layer,mask):
    a=layer.self_attn
    b,n,d=h.shape
    q,k,v=F.linear(h,a.in_proj_weight,a.in_proj_bias).view(b,n,3,a.num_heads,d//a.num_heads).permute(2,0,3,1,4).unbind(0)
    z=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
    return layer.norm1(h+a.out_proj(z.transpose(1,2).reshape(b,n,d)),None)


def original_ffn(layer,h):
    return layer.linear2(layer.activation(layer.linear1(h)))


@torch.no_grad()
def logits(teacher,e,replacements,device):
    h,mask,target=input_state(teacher,e,device)
    for i,layer in enumerate(teacher.h.layers):
        h=advance(h,layer,mask,replacements.get(i))
    if teacher.h.norm is not None:
        h=teacher.h.norm(h)
    return teacher.ar_predict_layer(h[:,e["x"].size(1)-1:]),target


@torch.no_grad()
def evaluate(teacher,rows,replacements,device,rank,world):
    sums=torch.zeros(10,dtype=torch.float64,device=device)
    for e in rows[rank::world]:
        z,tar=logits(teacher,e,replacements,device)
        p=e["prompt"]
        t=e["teacher"].to(device)
        target=e["target"].to(device)
        assert torch.equal(tar[:,p:],target)
        sums[:5]+=scores(z[:,p:],t,target).to(device)
        sums[5:]+=scores(t,t,target).to(device)
    dist.all_reduce(sums)
    a,k,g,c,n,ta,tk,tg,tc,tn=sums.tolist()
    assert n==tn and n>0
    return {"ce":a/n,"teacher_ce":ta/n,"ce_ratio":a/ta,"kl":k/n,
            "top1_agreement":g/n,"target_accuracy":c/n,"teacher_accuracy":tc/n,"tokens":int(n)}


def state_cpu(module):
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def score(m):
    return m["kl"]+.1*max(0.,m["ce"]-m["teacher_ce"])


@torch.no_grad()
def rollout(teacher,e,replacements,device,seed):
    x,b,y=(e[k].to(device) for k in ("x","b","y"))
    prompt=y[:,:e["prompt"]]
    y=prompt
    g=torch.Generator(device=device).manual_seed(seed)
    started=time.monotonic()
    for i in range(384):
        z,_=logits(teacher,{"x":x,"b":b,"y":y},replacements,device)
        v=sample_next(z[:,-1],y,eos=teacher.EOS,allow_eos=i>=10,top_k=15,
                      repetition_penalty=1.35,generator=g)
        if v.item()==teacher.EOS:
            break
        y=torch.cat((y,v),1)
    torch.cuda.synchronize(device)
    return y[:,prompt.size(1):],time.monotonic()-started


def train(args):
    from tools.train_s1_next import load_teacher,teacher_logits
    rank=int(os.environ["RANK"])
    local=int(os.environ["LOCAL_RANK"])
    world=int(os.environ["WORLD_SIZE"])
    assert world==2 and torch.cuda.device_count()==2
    torch.cuda.set_device(local)
    device=torch.device("cuda",local)
    dist.init_process_group("nccl",timeout=timedelta(minutes=10))
    try:
        started=time.monotonic()
        torch.manual_seed(2028)
        teacher,cfg,_=load_teacher(args.teacher,device)
        teacher.eval().requires_grad_(False)
        source=torch.load(args.first,map_location="cpu",weights_only=True)
        assert source["format"]=="s1-progressive-ffn-v1" and source["layer"]==0
        assert source["dim"]==cfg.dim and source["width"]==cfg.ff_dim and passed(source["validation"])
        ffn=SwiFFN(cfg.dim,cfg.ff_dim).to(device).eval()
        ffn.load_state_dict(source["state_dict"],strict=True)
        ffn.requires_grad_(False)
        accepted={0:ffn}
        valid=read_shards(args.data,world,"validation")
        assert len(valid)==24
        for e in valid[rank::world]:
            x,b,y=(e[k].to(device) for k in ("x","b","y"))
            t,tar=teacher_logits(teacher,x,b,y)
            z,_=logits(teacher,e,{},device)
            torch.testing.assert_close(t,z,rtol=3e-3,atol=3e-3)
            torch.testing.assert_close(t[:,e["prompt"]:],e["teacher"].to(device),rtol=3e-3,atol=3e-3)
        current=evaluate(teacher,valid,accepted,device,rank,world)
        assert passed(current),"Round 9 checkpoint failed cumulative validation"
        initial=current
        rows=read_shards(args.data,world,"train")
        random.Random(2028).shuffle(rows)
        rows=rows[:args.pairs]
        assert len(rows)==args.pairs
        # Each rank retains only its own prefix states, not all 24 layers.
        masks=[]; original=[]; hybrid=[]
        with torch.no_grad():
            for e in rows[rank::world]:
                h,m,_=input_state(teacher,e,device)
                masks.append(m)
                original.append(advance(h,teacher.h.layers[0],m))
                hybrid.append(advance(h,teacher.h.layers[0],m,accepted[0]))
        del rows
        history=[]
        tokens=0
        stop="all_layers_attempted"
        if rank==0:
            args.output.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        def persist():
            if rank!=0:
                return
            save_atomic({"format":"s1-progressive-stack-v1","dim":cfg.dim,"width":cfg.ff_dim,
                         "depth":cfg.depth,"replacements":{i:state_cpu(m) for i,m in accepted.items()},
                         "validation":current},args.output/"stack-best.pt")
            report={"world_size":world,"precision":"fp32","training_pairs":args.pairs,
                    "validation_pairs":len(valid),"initial":initial,"selected":current,
                    "accepted_layers":sorted(accepted),"rejected_layers":[r["layer"] for r in history if not r["accepted"]],
                    "history":history,"tokens":tokens,"stop_reason":stop,"seconds":time.monotonic()-started}
            tmp=args.output/"metrics.tmp"
            tmp.write_text(json.dumps(report,indent=2,allow_nan=False))
            os.replace(tmp,args.output/"metrics.json")
        def expired():
            flag=torch.tensor(int(time.monotonic()-started>=args.seconds),device=device)
            dist.broadcast(flag,src=0)
            return bool(flag.item())
        persist()
        for i in range(1,cfg.depth):
            if expired():
                stop="time_budget"
                break
            layer=teacher.h.layers[i]
            ffn=SwiFFN(cfg.dim,cfg.ff_dim).to(device)
            ffn.warm_start(layer)
            candidate={**accepted,i:ffn}
            warm=evaluate(teacher,valid,candidate,device,rank,world)
            best=None
            best_m=None
            best_epoch=None
            streak=int(passed(warm))
            layer_history=[{"epoch":0,"validation":warm,"train_loss":None}]
            # Warm start alone is not enough: require a trained passing checkpoint.
            hs=[]; fs=[]; posts=[]; next_original=[]
            with torch.no_grad():
                for a,b,m in zip(hybrid,original,masks):
                    h=pre_ffn(a,layer,m)
                    f=original_ffn(layer,h)
                    o=advance(b,layer,m)
                    hs.append(h.squeeze(0));fs.append(f.squeeze(0));posts.append(o.squeeze(0))
                    next_original.append(o)
            h=torch.cat(hs);f=torch.cat(fs);post=torch.cat(posts)
            del hs,fs,posts
            original=next_original
            num=torch.tensor(len(h),device=device)
            dist.all_reduce(num,op=dist.ReduceOp.MIN)
            steps=int(num.item())//args.batch
            assert steps>0
            model=DDP(ffn,device_ids=[local],broadcast_buffers=False)
            opt=torch.optim.AdamW(ffn.parameters(),lr=args.lr,weight_decay=.001,fused=True)
            for epoch in range(1,args.epochs+1):
                order=torch.randperm(len(h),device=device)
                ls=torch.zeros((),device=device)
                model.train()
                for j in range(steps):
                    idx=order[j*args.batch:(j+1)*args.batch]
                    a,t,o=h[idx],f[idx],post[idx]
                    pred=model(a)
                    # Local operator fit plus original full-prefix post-norm recovery.
                    loss=(pred-t).square().mean()/t.square().mean().clamp_min(1e-5)+F.mse_loss(layer.norm2(a+pred,None),o)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ffn.parameters(),1.,error_if_nonfinite=True)
                    opt.step()
                    ls+=loss.detach()
                    tokens+=args.batch*world
                dist.all_reduce(ls)
                model.eval()
                m=evaluate(teacher,valid,candidate,device,rank,world)
                streak=streak+1 if passed(m) else 0
                layer_history.append({"epoch":epoch,"validation":m,"train_loss":float(ls/(steps*world))})
                if passed(m) and (best_m is None or score(m)<score(best_m)):
                    best=state_cpu(ffn);best_m=m;best_epoch=epoch
                if rank==0:
                    print(json.dumps({"event":"layer_epoch","layer":i,"epoch":epoch,"tokens":tokens,
                                      "validation":m,"gate_streak":streak},allow_nan=False),flush=True)
                if streak>=2 or expired():
                    break
            # Conservative rollback: two consecutive passing checks, at least one after training.
            ok=streak>=2 and best is not None
            if ok:
                ffn.load_state_dict(best,strict=True)
                ffn.requires_grad_(False)
                accepted[i]=ffn.eval()
                current=best_m
            with torch.no_grad():
                hybrid=[advance(a,layer,m,accepted.get(i)) for a,m in zip(hybrid,masks)]
            history.append({"layer":i,"accepted":ok,"selected_epoch":best_epoch if ok else None,
                            "warm":warm,"selected":current,"epochs":layer_history})
            del model,opt,h,f,post,candidate
            if not ok:
                del ffn
            persist()
            if rank==0:
                print(json.dumps({"event":"layer_result","layer":i,"accepted":ok,
                                  "accepted_count":len(accepted),"validation":current},allow_nan=False),flush=True)
            dist.barrier()
        # Independent of the training decision: re-evaluate the saved cumulative model.
        final=evaluate(teacher,valid,accepted,device,rank,world)
        assert passed(final)
        assert abs(final["ce"]-current["ce"])<1e-5
        current=final
        persist()
        # Bounded same-policy free-run diagnostic on four fixed validation prompts.
        trial=[]
        for e in valid[:4][rank::world]:
            seed=40100+e["index"]
            t,ts=rollout(teacher,e,{},device,seed)
            z,zs=rollout(teacher,e,accepted,device,seed)
            common=min(t.size(1),z.size(1))
            equal=(t[:,:common]==z[:,:common]).sum().item()
            trial.append({"index":e["index"],"teacher_tokens":t.size(1),"candidate_tokens":z.size(1),
                          "positional_matches":equal,"compared":common,"teacher_seconds":ts,"candidate_seconds":zs})
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
    from types import SimpleNamespace
    torch.manual_seed(3)
    class Norm(nn.LayerNorm):
        def forward(self,x,unused=None):
            return super().forward(x)
    layer=SimpleNamespace(self_attn=nn.MultiheadAttention(32,4,batch_first=True),
                          norm1=Norm(32),norm2=Norm(32),linear1=nn.Linear(32,64),
                          linear2=nn.Linear(64,32),activation=F.relu)
    h=torch.randn(1,11,32)
    m=prefix_mask(5,6,h.device)
    z=pre_ffn(h,layer,m)
    torch.testing.assert_close(layer.norm2(z+original_ffn(layer,z)),advance(h,layer,m))
    ffn=SwiFFN(32,64)
    ffn.warm_start(layer,scale=20.)
    torch.testing.assert_close(advance(h,layer,m,ffn),layer.norm2(z+ffn(z)))
    assert passed({"ce_ratio":1.,"kl":.001,"top1_agreement":.95})
    assert not passed({"ce_ratio":1.,"kl":.03,"top1_agreement":.95})
    print("Progressive stack prefix, replacement and cumulative gate checks passed")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    for name in ("teacher","data","first","output"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--pairs",type=int,default=384)
    p.add_argument("--batch",type=int,default=1024)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--seconds",type=int,default=1800)
    args=p.parse_args()
    if args.selftest:
        selftest()
    else:
        train(args)
