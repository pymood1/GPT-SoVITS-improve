"""Round 13: static KV cache for original post-LN and migrated SwiGLU S1.
No training; fixed teacher-forced trajectories compare per-token output and latency.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from AR.models.s1_next import prefix_mask, sample_next
from tools.audit_s1_next_transfer import norm, block
from tools.train_s1_progressive_stack import logits
from tools.train_s1_progressive_ffn import SwiFFN
from tools.train_s1_next_ddp import read_shards


class Cache:
    def __init__(self,teacher,mods,x,b,prompt,capacity):
        self.teacher=teacher
        self.mods=mods
        self.n=x.size(1)
        self.m=prompt.size(1)
        self.length=self.n+self.m
        self.capacity=self.length+capacity
        t=teacher.ar_text_position(teacher.ar_text_embedding(x)+teacher.bert_proj(b.transpose(1,2)))
        p=teacher.ar_audio_position(teacher.ar_audio_embedding(prompt))
        h=torch.cat((t,p),1)
        self.kv=[]
        mask=prefix_mask(self.n,self.m,x.device)
        # Reserve the sinusoidal table, then use the exact original positional formula.
        self.position=teacher.ar_audio_position
        assert not self.position.reverse
        self.position.extend_pe(h.new_empty((1,self.m+capacity,1)))
        self.z=self.run(h,mask,prefill=True)

    def run(self,h,mask,prefill=False):
        b,n,d=h.shape
        end=self.length if prefill else self.length+1
        for i,l in enumerate(self.teacher.h.layers):
            a=l.self_attn
            q,k,v=F.linear(h,a.in_proj_weight,a.in_proj_bias).view(b,n,3,a.num_heads,d//a.num_heads).permute(2,0,3,1,4).unbind(0)
            if prefill:
                shape=(b,a.num_heads,self.capacity,d//a.num_heads)
                keys=k.new_empty(shape);values=v.new_empty(shape)
                self.kv.append((keys,values))
                start=0
            else:
                keys,values=self.kv[i]
                start=self.length
            keys[:,:,start:end].copy_(k)
            values[:,:,start:end].copy_(v)
            z=F.scaled_dot_product_attention(q,keys[:,:,:end],values[:,:,:end],attn_mask=mask)
            h=norm(h+a.out_proj(z.transpose(1,2).reshape(b,n,d)),l.norm1,False)
            f=self.mods.get(i)
            z=l.linear2(l.activation(l.linear1(h))) if f is None else f(h)
            h=norm(h+z,l.norm2,False)
        if self.teacher.h.norm is not None:
            h=self.teacher.h.norm(h)
        return self.teacher.ar_predict_layer(h[:,-1])

    def step(self,token):
        assert self.length<self.capacity
        index=self.length-self.n
        emb=self.teacher.ar_audio_embedding(token)
        h=self.position.dropout(emb*self.position.x_scale+self.position.alpha*self.position.pe[:,index:index+1])
        self.z=self.run(h,None)
        self.length+=1
        return self.z

    def bytes(self):
        return sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in self.kv)


@torch.inference_mode()
def full_trace(teacher,mods,x,b,prompt,tokens):
    out=[]
    y=prompt
    for token in tokens.split(1,dim=1):
        z,_=logits(teacher,{"x":x,"b":b,"y":y},mods,x.device)
        out.append(z[:,-1].clone())
        y=torch.cat((y,token),1)
    return torch.cat(out,0)


@torch.inference_mode()
def cache_trace(teacher,mods,x,b,prompt,tokens):
    c=Cache(teacher,mods,x,b,prompt,tokens.size(1))
    out=[]
    for i,token in enumerate(tokens.split(1,dim=1)):
        out.append(c.z.clone())
        if i+1<tokens.size(1):
            c.step(token)
    return torch.cat(out,0),c.bytes()


@torch.inference_mode()
def generate(teacher,mods,x,b,prompt,seed,cap,cached):
    y=prompt
    g=torch.Generator(device=x.device).manual_seed(seed)
    c=Cache(teacher,mods,x,b,prompt,cap) if cached else None
    for i in range(cap):
        if cached:
            z=c.z
        else:
            z,_=logits(teacher,{"x":x,"b":b,"y":y},mods,x.device)
            z=z[:,-1]
        token=sample_next(z,y,eos=teacher.EOS,allow_eos=i>=10,top_k=15,
                          repetition_penalty=1.35,generator=g)
        if token.item()==teacher.EOS:
            break
        y=torch.cat((y,token),1)
        if cached and i+1<cap:
            c.step(token)
    return y[:,prompt.size(1):]


def load_stack(teacher,path,device):
    saved=torch.load(path,map_location="cpu",weights_only=True)
    assert saved["format"]=="s1-progressive-stack-v1"
    assert saved["depth"]==len(teacher.h.layers)==24
    assert sorted(saved["replacements"])==list(range(24))
    mods={}
    for i,state in saved["replacements"].items():
        m=SwiFFN(saved["dim"],saved["width"]).to(device)
        m.load_state_dict(state,strict=True)
        mods[i]=m.eval().requires_grad_(False)
    return mods


def timed(fn,device):
    torch.cuda.synchronize(device)
    start=time.perf_counter()
    result=fn()
    torch.cuda.synchronize(device)
    return result,time.perf_counter()-start


@torch.inference_mode()
def benchmark(a):
    from tools.train_s1_next import load_teacher,teacher_logits
    device=torch.device("cuda:0")
    teacher,_,_=load_teacher(a.teacher,device)
    mods=load_stack(teacher,a.stack,device)
    rows=read_shards(a.data,2,"validation")
    assert len(rows)==24
    rows=rows[a.rank::2]
    a.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for e in rows:
        x,b,y=(e[k].to(device) for k in ("x","b","y"))
        p=e["prompt"]
        prompt=y[:,:p]
        tokens=y[:,p:p+128]
        assert tokens.size(1)>=12
        original,_=teacher_logits(teacher,x,b,y)
        torch.testing.assert_close(original[:,p:],e["teacher"].to(device),rtol=3e-3,atol=3e-3)
        # Unmeasured warmup, then alternating matched fixed-token comparisons.
        for m in ({},mods):
            full_trace(teacher,m,x,b,prompt,tokens[:,:2])
            cache_trace(teacher,m,x,b,prompt,tokens[:,:2])
        row={"index":e["index"],"tokens":tokens.size(1),"prompt_tokens":p,"models":{}}
        for label,m in (("teacher",{}),("student",mods)):
            if e["index"]%2:
                (c,mem),ct=timed(lambda:cache_trace(teacher,m,x,b,prompt,tokens),device)
                f,ft=timed(lambda:full_trace(teacher,m,x,b,prompt,tokens),device)
            else:
                f,ft=timed(lambda:full_trace(teacher,m,x,b,prompt,tokens),device)
                (c,mem),ct=timed(lambda:cache_trace(teacher,m,x,b,prompt,tokens),device)
            delta=(f-c).abs()
            close=bool(torch.allclose(f,c,rtol=3e-3,atol=3e-3))
            row["models"][label]={"full_seconds":ft,"cached_seconds":ct,
                "max_abs":float(delta.max()),"mean_abs":float(delta.mean()),"parity":close,
                "top1_matches":int((f.argmax(-1)==c.argmax(-1)).sum()),"cache_bytes":mem}
            if e["index"]<4:
                seed=40100+e["index"]
                g,gt=timed(lambda:generate(teacher,m,x,b,prompt,seed,384,False),device)
                h,ht=timed(lambda:generate(teacher,m,x,b,prompt,seed,384,True),device)
                row["models"][label].update(sampled_equal=bool(torch.equal(g,h)),
                    sampled_full_tokens=g.size(1),sampled_cached_tokens=h.size(1),
                    sampled_full_seconds=gt,sampled_cached_seconds=ht)
        results.append(row)
        print(json.dumps({"event":"cache_trial",**row},allow_nan=False),flush=True)
        (a.output/"metrics.json").write_text(json.dumps(results,allow_nan=False))
    assert len(results)==12


def selftest():
    from types import SimpleNamespace
    class Pos(nn.Module):
        def __init__(self,d):
            super().__init__()
            self.register_buffer("pe",torch.randn(1,128,d))
            self.alpha=nn.Parameter(torch.ones(1))
            self.x_scale=1.;self.reverse=False;self.dropout=nn.Identity()
        def extend_pe(self,x):
            assert x.size(1)<=self.pe.size(1)
        def forward(self,x):
            return x+self.alpha*self.pe[:,:x.size(1)]
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn=nn.MultiheadAttention(16,4,batch_first=True)
            self.norm1=nn.LayerNorm(16);self.norm2=nn.LayerNorm(16)
            self.linear1=nn.Linear(16,32);self.linear2=nn.Linear(32,16)
            self.activation=F.relu
    torch.manual_seed(31)
    t=SimpleNamespace(ar_text_embedding=nn.Embedding(32,16),bert_proj=nn.Linear(8,16),
        ar_text_position=Pos(16),ar_audio_embedding=nn.Embedding(32,16),ar_audio_position=Pos(16),
        h=SimpleNamespace(layers=[Layer(),Layer()],norm=None),ar_predict_layer=nn.Linear(16,32))
    x=torch.randint(32,(1,7));b=torch.randn(1,8,7);p=torch.randint(32,(1,5));y=torch.randint(32,(1,9))
    ffn=SwiFFN(16,32);ffn.warm_start(t.h.layers[1],scale=8.)
    with torch.inference_mode():
        for mods in ({},{1:ffn}):
            c=Cache(t,mods,x,b,p,9)
            context=p
            for token in y.split(1,dim=1):
                h=torch.cat((t.ar_text_position(t.ar_text_embedding(x)+t.bert_proj(b.transpose(1,2))),
                             t.ar_audio_position(t.ar_audio_embedding(context))),1)
                mask=prefix_mask(x.size(1),context.size(1),x.device)
                for i,l in enumerate(t.h.layers):
                    h=block(h,l,mask,"identity",mods.get(i))
                torch.testing.assert_close(c.z,t.ar_predict_layer(h[:,-1]),rtol=1e-5,atol=1e-5)
                c.step(token)
                context=torch.cat((context,token),1)
    print("Static KV cache parity passed for original and SwiGLU blocks")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    for name in ("teacher","data","stack","output"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--rank",type=int,default=0)
    a=p.parse_args()
    if a.selftest:
        selftest()
    else:
        benchmark(a)
