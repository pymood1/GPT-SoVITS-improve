"""Round 15: polish the accepted RMS/SwiGLU checkpoint; cache parity and speed.
All original attention, position and output weights stay frozen.
Development selection never sees the new waveform test set.
"""
import argparse
import contextlib
import json
import os
import random
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from AR.models.s1_next import prefix_mask, sample_next
from tools.train_s1_affine_rms import HybridRMS, AffineRMS, evaluate, snap, restore, passed
from tools.train_s1_fullwidth_ffn import loss_fn
from tools.train_s1_next_ddp import read_shards, save_atomic, batches, gradient_multiplier
from tools.audit_s1_stack_kv import Cache, load_stack, timed
from tools.audit_s1_next_transfer import norm
from tools.train_s1_progressive_stack import logits


def load_rms(teacher,stack,checkpoint,device):
    mods=load_stack(teacher,stack,device)
    m=HybridRMS(teacher,mods).to(device).eval()
    s=torch.load(checkpoint,map_location="cpu",weights_only=True)
    assert s["format"]=="s1-affine-rms-v1" and s["alpha"]==1.
    assert s["depth"]==len(teacher.h.layers)==24 and s["width"]==mods[0].up.out_features
    assert passed(s["validation"])
    restore(m,s["state"],1.)
    return m


def improves(v,b):
    return (passed(v) and v["ce_ratio"]<=b["ce_ratio"]+.0005
            and v["kl"]<=b["kl"] and v["top1_agreement"]>=b["top1_agreement"])


def train(a):
    from tools.train_s1_next import load_teacher,teacher_logits
    rank=int(os.environ["RANK"]);local=int(os.environ["LOCAL_RANK"]);world=int(os.environ["WORLD_SIZE"])
    assert world==2 and torch.cuda.device_count()==2
    torch.cuda.set_device(local)
    device=torch.device("cuda",local)
    dist.init_process_group("nccl",timeout=timedelta(minutes=10))
    try:
        started=time.monotonic()
        torch.manual_seed(2035)
        teacher,cfg,_=load_teacher(a.teacher,device)
        teacher.eval().requires_grad_(False)
        m=load_rms(teacher,a.stack,a.resume,device)
        m.ffns.requires_grad_(True);m.norms.requires_grad_(True)
        rows=read_shards(a.data,world,"train")
        random.Random(2035).shuffle(rows);rows=rows[:a.pairs]
        valid=read_shards(a.data,world,"validation")
        assert len(rows)==a.pairs and len(valid)==24
        for e in valid[rank::world]:
            x,b,y=(e[k].to(device) for k in ("x","b","y"))
            t,_=teacher_logits(teacher,x,b,y)
            torch.testing.assert_close(t[:,e["prompt"]:],e["teacher"].to(device),rtol=3e-3,atol=3e-3)
        baseline=evaluate(m,valid,device,rank,world)
        saved=torch.load(a.resume,map_location="cpu",weights_only=True)
        assert abs(baseline["ce"]-saved["validation"]["ce"])<1e-5
        del saved
        best=snap(m);selected=baseline;selected_epoch=0
        history=[];tokens=0;steps=0;stale=0;stop="epoch_budget"
        if rank==0:
            a.output.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        model=DDP(m,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True)
        params=list(m.ffns.parameters())+list(m.norms.parameters())
        opt=torch.optim.AdamW([{"params":m.ffns.parameters(),"lr":2e-6},
                               {"params":m.norms.parameters(),"lr":5e-5}],weight_decay=0.,fused=True)
        def persist():
            if rank==0:
                save_atomic({"format":"s1-affine-rms-v1","dim":cfg.dim,"depth":cfg.depth,
                    "width":m.ffns[0].up.out_features,"alpha":1.,"state":best,
                    "validation":selected},a.output/"norm-best.pt")
                report={"world_size":world,"precision":"fp32","training_pairs":len(rows),
                    "validation_pairs":len(valid),"baseline":baseline,"selected":selected,
                    "selected_epoch":selected_epoch,"target_met":selected["top1_agreement"]>=.98,
                    "history":history,"tokens":tokens,"steps":steps,"stop_reason":stop,
                    "seconds":time.monotonic()-started}
                tmp=a.output/"metrics.tmp";tmp.write_text(json.dumps(report,indent=2,allow_nan=False))
                os.replace(tmp,a.output/"metrics.json")
        def expired():
            flag=torch.tensor(int(time.monotonic()-started>=a.seconds),device=device)
            dist.broadcast(flag,src=0)
            return bool(flag.item())
        persist()
        for epoch in range(1,a.epochs+1):
            order=list(range(len(rows)));random.Random(2035+epoch).shuffle(order)
            totals=torch.zeros(2,device=device,dtype=torch.float64);exhausted=False
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
                    exhausted=True;break
            dist.all_reduce(totals)
            v=evaluate(m,valid,device,rank,world)
            accepted=improves(v,baseline) and (-v["top1_agreement"],v["kl"],v["ce"])<(-selected["top1_agreement"],selected["kl"],selected["ce"])
            if accepted:
                selected=v;best=snap(m);selected_epoch=epoch;stale=0
            else:
                stale+=1
            history.append({"epoch":epoch,"validation":v,"accepted":accepted,
                            "train_loss":float(totals[0]/totals[1]),"partial_epoch":exhausted})
            if rank==0:
                print(json.dumps({"event":"layer_epoch","epoch":epoch,"tokens":tokens,
                                  "validation":v,"accepted":accepted}),flush=True)
            persist();dist.barrier()
            if selected["top1_agreement"]>=.98:
                stop="target_met";break
            if exhausted or expired():
                stop="time_budget";break
            if stale>=3:
                stop="no_improvement";break
            if stale==2:
                restore(m,best,1.)
                opt.state.clear()
                for g in opt.param_groups:
                    g["lr"]*=.5
        restore(m,best,1.)
        final=evaluate(m,valid,device,rank,world)
        assert improves(final,baseline) and abs(final["ce"]-selected["ce"])<1e-5
        selected=final;persist();dist.barrier()
    finally:
        dist.destroy_process_group()


class RMSCache(Cache):
    def __init__(self,teacher,mods,norms,x,b,prompt,capacity):
        self.norms=norms
        super().__init__(teacher,mods,x,b,prompt,capacity)

    def run(self,h,mask,prefill=False):
        b,n,d=h.shape
        end=self.length if prefill else self.length+1
        for i,l in enumerate(self.teacher.h.layers):
            a=l.self_attn
            q,k,v=F.linear(h,a.in_proj_weight,a.in_proj_bias).view(b,n,3,a.num_heads,d//a.num_heads).permute(2,0,3,1,4).unbind(0)
            if prefill:
                shape=(b,a.num_heads,self.capacity,d//a.num_heads)
                keys=k.new_empty(shape);values=v.new_empty(shape)
                self.kv.append((keys,values));start=0
            else:
                keys,values=self.kv[i];start=self.length
            keys[:,:,start:end].copy_(k);values[:,:,start:end].copy_(v)
            z=F.scaled_dot_product_attention(q,keys[:,:,:end],values[:,:,:end],attn_mask=mask)
            h=h+a.out_proj(z.transpose(1,2).reshape(b,n,d))
            h=norm(h,l.norm1,False) if self.norms is None else self.norms[2*i](h)
            f=self.mods.get(i)
            h=h+(l.linear2(l.activation(l.linear1(h))) if f is None else f(h))
            h=norm(h,l.norm2,False) if self.norms is None else self.norms[2*i+1](h)
        if self.teacher.h.norm is not None:
            h=self.teacher.h.norm(h)
        return self.teacher.ar_predict_layer(h[:,-1])


class Generator:
    def __init__(self,teacher,mods,rms=None):
        self.teacher=teacher;self.mods=mods;self.rms=rms
        self.norms=None if rms is None else rms.norms
        self.cfg=SimpleNamespace(codes=teacher.EOS)

    @torch.inference_mode()
    def full(self,x,b,y):
        if self.rms is not None:
            return self.rms({"x":x,"b":b,"y":y,"prompt":y.size(1)})[:,-1]
        z,_=logits(self.teacher,{"x":x,"b":b,"y":y},self.mods,x.device)
        return z[:,-1]

    @torch.inference_mode()
    def generate(self,x,b,prompt,*,max_new,min_new,top_k,repetition_penalty,generator,cached=True):
        y=prompt
        c=RMSCache(self.teacher,self.mods,self.norms,x,b,prompt,max_new) if cached else None
        for i in range(max_new):
            z=c.z if cached else self.full(x,b,y)
            token=sample_next(z,y,eos=self.teacher.EOS,allow_eos=i>=min_new,top_k=top_k,
                              repetition_penalty=repetition_penalty,generator=generator)
            if token.item()==self.teacher.EOS:
                break
            y=torch.cat((y,token),1)
            if cached and i+1<max_new:
                c.step(token)
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        return y[:,prompt.size(1):],None


def generators(teacher,stack,original,refined,device):
    base=load_stack(teacher,stack,device)
    prev=load_rms(teacher,stack,original,device).requires_grad_(False)
    final=load_rms(teacher,stack,refined,device).requires_grad_(False)
    return {"teacher":Generator(teacher,{}),"round12":Generator(teacher,base),
            "round14":Generator(teacher,dict(enumerate(prev.ffns)),prev),
            "student":Generator(teacher,dict(enumerate(final.ffns)),final)}


@torch.inference_mode()
def trace(g,x,b,prompt,tokens,cached):
    y=prompt
    c=RMSCache(g.teacher,g.mods,g.norms,x,b,prompt,tokens.size(1)) if cached else None
    out=[]
    for i,token in enumerate(tokens.split(1,dim=1)):
        out.append((c.z if cached else g.full(x,b,y)).clone())
        if i+1<tokens.size(1):
            if cached:
                c.step(token)
            else:
                y=torch.cat((y,token),1)
    return torch.cat(out,0),c.bytes() if cached else 0


@torch.inference_mode()
def benchmark(a):
    from tools.train_s1_next import load_teacher
    device=torch.device("cuda:0")
    t,_,_=load_teacher(a.teacher,device)
    gs=generators(t,a.stack,a.resume,a.refined,device)
    rows=read_shards(a.data,2,"validation")[a.rank::2]
    assert len(rows)==12
    a.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for e in rows:
        x,b,y=(e[k].to(device) for k in ("x","b","y"));p=e["prompt"];prompt=y[:,:p]
        tokens=y[:,p:p+128]
        row={"index":e["index"],"tokens":tokens.size(1),"models":{}}
        for label,g in gs.items():
            trace(g,x,b,prompt,tokens[:,:2],False);trace(g,x,b,prompt,tokens[:,:2],True)
            if e["index"]%2:
                (c,mem),ct=timed(lambda:trace(g,x,b,prompt,tokens,True),device)
                (f,_),ft=timed(lambda:trace(g,x,b,prompt,tokens,False),device)
            else:
                (f,_),ft=timed(lambda:trace(g,x,b,prompt,tokens,False),device)
                (c,mem),ct=timed(lambda:trace(g,x,b,prompt,tokens,True),device)
            delta=(f-c).abs()
            r={"full_seconds":ft,"cached_seconds":ct,"max_abs":float(delta.max()),
                "mean_abs":float(delta.mean()),"parity":bool(torch.allclose(f,c,rtol=3e-3,atol=3e-3)),
                "top1_matches":int((f.argmax(-1)==c.argmax(-1)).sum()),"cache_bytes":mem}
            if e["index"]<4:
                def gen(cached):
                    return g.generate(x,b,prompt,max_new=384,min_new=10,top_k=15,
                        repetition_penalty=1.35,generator=torch.Generator(device=device).manual_seed(40100+e["index"]),
                        cached=cached)[0]
                f,ft=timed(lambda:gen(False),device);c,ct=timed(lambda:gen(True),device)
                r.update(sampled_equal=bool(torch.equal(f,c)),sampled_full_seconds=ft,
                         sampled_cached_seconds=ct,sampled_full_tokens=f.size(1),sampled_cached_tokens=c.size(1))
            row["models"][label]=r
        results.append(row)
        print(json.dumps({"event":"cache_trial",**row},allow_nan=False),flush=True)
        (a.output/"metrics.json").write_text(json.dumps(results,allow_nan=False))


def selftest():
    from tools.train_s1_affine_rms import selftest as norm_test
    from tools.audit_s1_stack_kv import selftest as cache_test
    norm_test();cache_test()
    class Pos(nn.Module):
        def __init__(self):
            super().__init__();self.register_buffer("pe",torch.randn(1,128,16))
            self.alpha=nn.Parameter(torch.ones(1));self.x_scale=1.;self.reverse=False;self.dropout=nn.Identity()
        def extend_pe(self,x):
            assert x.size(1)<=self.pe.size(1)
        def forward(self,x):
            return x+self.alpha*self.pe[:,:x.size(1)]
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn=nn.MultiheadAttention(16,4,batch_first=True)
            self.norm1=nn.LayerNorm(16);self.norm2=nn.LayerNorm(16)
            self.linear1=nn.Linear(16,32);self.linear2=nn.Linear(32,16);self.activation=F.relu
    from tools.train_s1_progressive_ffn import SwiFFN
    torch.manual_seed(35)
    t=SimpleNamespace(ar_text_embedding=nn.Embedding(32,16),bert_proj=nn.Linear(8,16),
        ar_text_position=Pos(),ar_audio_embedding=nn.Embedding(32,16),ar_audio_position=Pos(),
        h=SimpleNamespace(layers=[Layer(),Layer()],norm=None),ar_predict_layer=nn.Linear(16,32))
    mods={i:SwiFFN(16,32) for i in range(2)}
    for i,f in mods.items():
        f.warm_start(t.h.layers[i],scale=8.)
    norms=[AffineRMS(n) for l in t.h.layers for n in (l.norm1,l.norm2)]
    for n in norms:n.alpha=1.
    x=torch.randint(32,(1,7));b=torch.randn(1,8,7);p=torch.randint(32,(1,5));y=torch.randint(32,(1,9))
    with torch.inference_mode():
        c=RMSCache(t,mods,norms,x,b,p,9);context=p
        for token in y.split(1,dim=1):
            h=torch.cat((t.ar_text_position(t.ar_text_embedding(x)+t.bert_proj(b.transpose(1,2))),
                         t.ar_audio_position(t.ar_audio_embedding(context))),1)
            mask=prefix_mask(x.size(1),context.size(1),x.device)
            for i,l in enumerate(t.h.layers):
                a=l.self_attn;bs,n,d=h.shape
                q,k,v=F.linear(h,a.in_proj_weight,a.in_proj_bias).view(bs,n,3,4,4).permute(2,0,3,1,4).unbind(0)
                z=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
                h=norms[2*i](h+a.out_proj(z.transpose(1,2).reshape(bs,n,d)))
                h=norms[2*i+1](h+mods[i](h))
            torch.testing.assert_close(c.z,t.ar_predict_layer(h[:,-1]),rtol=1e-5,atol=1e-5)
            c.step(token);context=torch.cat((context,token),1)
    print("RMS/SwiGLU static-cache parity passed")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    p.add_argument("--mode",choices=("train","benchmark"),default="train")
    for name in ("teacher","data","stack","resume","refined","output"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--rank",type=int,default=0)
    p.add_argument("--pairs",type=int,default=512)
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--seconds",type=int,default=720)
    a=p.parse_args()
    if a.selftest:selftest()
    elif a.mode=="train":train(a)
    else:benchmark(a)
