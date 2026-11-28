"""Frozen S1 transfer audit: isolate architecture changes on identical cached prompts.
No training, no production changes, no raw text/audio in reports.
"""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from AR.models.s1_next import S1Next, prefix_mask, rope_terms, rotary


MODES=("identity","no_abs_pos","rope","post_rms","pre_ln","fresh_ffn","trained_ffn")


def norm(x,n,rms):
    if not rms:
        return F.layer_norm(x,n.normalized_shape,n.weight,n.bias,n.eps)
    return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+n.eps)*n.weight


def block(x,layer,mask,mode,ffn=None):
    a=layer.self_attn
    b,n,d=x.shape
    h=a.num_heads
    pre=mode=="pre_ln"
    rms=mode=="post_rms"
    z=norm(x,layer.norm1,rms) if pre else x
    q,k,v=F.linear(z,a.in_proj_weight,a.in_proj_bias).view(b,n,3,h,d//h).permute(2,0,3,1,4).unbind(0)
    if mode=="rope":
        p=torch.arange(n,device=x.device).expand(b,-1)
        terms=rope_terms(p,d//h,10000.)
        q,k=rotary(q,terms),rotary(k,terms)
    z=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
    x=x+a.out_proj(z.transpose(1,2).reshape(b,n,d))
    if not pre:
        x=norm(x,layer.norm1,rms)
    z=norm(x,layer.norm2,rms) if pre else x
    if ffn is None:
        z=layer.linear2(layer.activation(layer.linear1(z)))
    else:
        z=ffn.down(F.silu(ffn.gate(z))*ffn.up(z))
    x=x+z
    return x if pre else norm(x,layer.norm2,rms)


@torch.no_grad()
def variant(teacher,x,b,y,mode,student=None,only=None):
    device=x.device
    n=x.size(1)
    xlen=torch.tensor([n],device=device)
    ylen=torch.tensor([y.size(1)],device=device)
    xy,_,target=teacher.make_input_data(x,xlen,y,ylen,b)
    if mode in ("no_abs_pos","rope"):
        if only is not None:
            raise ValueError("Position input changes must be global")
        xy=torch.cat((teacher.ar_text_embedding(x)+teacher.bert_proj(b.transpose(1,2)),
                      teacher.ar_audio_embedding(y)),dim=1)
    mask=prefix_mask(n,y.size(1),device)
    for i,layer in enumerate(teacher.h.layers):
        use=mode if only is None or i==only else "identity"
        ffn=student.blocks[i] if use in ("fresh_ffn","trained_ffn") else None
        xy=block(xy,layer,mask,use,ffn)
    if teacher.h.norm is not None:
        xy=teacher.h.norm(xy)
    return teacher.ar_predict_layer(xy[:,n-1:]),target


@torch.no_grad()
def cache_error(student,x,b,y,prompt):
    full,_=student(x,b,y)
    n=x.size(1)
    a=torch.cat((student.text(x)+student.bert(b.transpose(1,2)),student.audio(y[:,:prompt])),1)
    h,kv=student._run(a,0,prefix_mask(n,prompt,x.device))
    errors=[(student.head(h[:,-1])-full[:,prompt]).abs().max().item()]
    for i in range(prompt,min(y.size(1),prompt+12)):
        h,kv=student._run(student.audio(y[:,i:i+1]),n+i,None,kv)
        errors.append((student.head(h[:,0])-full[:,i+1]).abs().max().item())
    return max(errors)


def scores(z,t,y):
    z,t=z.float(),t.float()
    return torch.tensor([F.cross_entropy(z.flatten(0,1),y.flatten(),reduction="sum").item(),
       F.kl_div(F.log_softmax(z,-1),F.softmax(t,-1),reduction="sum").item(),
       (z.argmax(-1)==t.argmax(-1)).sum().item(),
       (z.argmax(-1)==y).sum().item(),y.numel()],dtype=torch.float64)


@torch.no_grad()
def audit(args):
    from tools.train_s1_next import load_teacher,teacher_logits
    device=torch.device(args.device)
    teacher,cfg,_=load_teacher(args.teacher,device)
    torch.manual_seed(1234)
    fresh=S1Next(cfg).to(device).eval()
    fresh.initialize_from_teacher(teacher)
    checkpoint=torch.load(args.student,map_location="cpu",weights_only=True)
    trained=S1Next(cfg).to(device).eval()
    trained.load_state_dict(checkpoint["state_dict"],strict=True)
    rows=[]
    for r in range(2):
        rows.extend(torch.load(args.data/f"pseudo{r}/validation.pt",map_location="cpu",weights_only=True))
    rows=rows[args.rank::args.world]
    totals={}
    identity_max=cache_max=0.
    for e in rows:
        x,b,y=(e[k].to(device) for k in ("x","b","y"))
        p=e["prompt"]
        t,tar=teacher_logits(teacher,x,b,y)
        target=tar[:,p:]
        assert torch.equal(target,e["target"].to(device))
        torch.testing.assert_close(t[:,p:],e["teacher"].to(device),rtol=3e-3,atol=3e-3)
        for name in ("teacher",*MODES,"fresh_next","trained_next","early_ffn","late_ffn"):
            if name=="teacher":
                z=t
            elif name in ("fresh_next","trained_next"):
                z,_=(fresh if name=="fresh_next" else trained)(x,b,y)
            else:
                only=0 if name=="early_ffn" else (cfg.depth-1 if name=="late_ffn" else None)
                mode="fresh_ffn" if only is not None else name
                z,_=variant(teacher,x,b,y,mode,trained if name=="trained_ffn" else fresh,only)
            if name=="identity":
                identity_max=max(identity_max,(z-t).abs().max().item())
                torch.testing.assert_close(z,t,rtol=3e-3,atol=3e-3)
            totals[name]=totals.get(name,torch.zeros(5,dtype=torch.float64))+scores(z[:,p:],t[:,p:],target)
        cache_max=max(cache_max,cache_error(trained,x,b,y,p))
    report={"rank":args.rank,"pairs":len(rows),"identity_max_abs":identity_max,
            "student_cache_max_abs":cache_max,"sums":{k:v.tolist() for k,v in totals.items()}}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))


def selftest():
    from types import SimpleNamespace
    torch.manual_seed(3)
    x=torch.randn(1,11,32)
    a=nn.MultiheadAttention(32,4,batch_first=True,dropout=0).eval()
    l=SimpleNamespace(self_attn=a,norm1=nn.LayerNorm(32),norm2=nn.LayerNorm(32),
         linear1=nn.Linear(32,64),linear2=nn.Linear(64,32),activation=F.relu)
    mask=prefix_mask(5,6,x.device)
    q=l.norm1(x+a(x,x,x,attn_mask=~mask,need_weights=False)[0])
    ref=l.norm2(q+l.linear2(l.activation(l.linear1(q))))
    torch.testing.assert_close(block(x,l,mask,"identity"),ref,rtol=1e-5,atol=2e-6)
    for mode in ("rope","post_rms","pre_ln"):
        z=block(x,l,mask,mode)
        assert z.shape==x.shape and torch.isfinite(z).all() and not torch.allclose(z,ref)
    print("Transfer audit identity/ablation selftest passed")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    for name in ("teacher","student","data","output"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--rank",type=int,default=0)
    p.add_argument("--world",type=int,default=1)
    args=p.parse_args()
    if args.selftest:
        selftest()
    else:
        audit(args)
