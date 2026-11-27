"""Research-only AR duration controls and pretrained masked semantic adaptation."""
import math
import torch
import torch.nn.functional as F

class PretrainedMasked:
    """Borrow native embeddings/transformer/head; masked input uses EOS embedding.
    Text and reference prefix cannot see target, target can attend bidirectionally.
    """
    def __init__(self,model,bert):self.model=model;self.bert=bert
    def __call__(self,r,masked,local=False):
        m=self.model
        x=m.ar_text_position(m.ar_text_embedding(r["x"])+m.bert_proj(self.bert(r).transpose(1,2)))
        audio=m.ar_audio_position(m.ar_audio_embedding(torch.cat((r["prompt"],masked),1)))
        n=x.size(1);p=r["prompt"].size(1);t=masked.size(1);prefix=n+p
        seq=torch.cat((x,audio),1)
        mask=torch.zeros((prefix+t,prefix+t),dtype=seq.dtype,device=seq.device)
        mask[:prefix,prefix:]=float("-inf")
        mask[:n,n:prefix]=float("-inf")
        # Match causal reference context; only target is bidirectional.
        mask[n:prefix,n:prefix].masked_fill_(torch.triu(torch.ones(p,p,device=seq.device,dtype=torch.bool),1),float("-inf"))
        if local:
            count=r["target"]["x"].size(1);start=n-count
            pos=(torch.arange(t,device=seq.device)+.5)*count/t
            distance=(pos[:,None]-(torch.arange(count,device=seq.device)+.5)[None]).abs()
            mask[prefix:,start:n].masked_fill_(distance>max(4.,count*.25),float("-inf"))
        h,_=m.h((seq,None),mask=mask)
        return m.ar_predict_layer(h[:,-t:])[...,:1024]

def duration_logits(logits,step,expected,mode,eos=1024):
    if mode=="native":return logits
    z=logits.clone()
    if mode=="soft":z[...,eos]+=4*math.tanh((step/max(1,expected)-1)*3)
    elif mode=="band":
        if step<max(11,round(.75*expected)):z[...,eos]=-float("inf")
        elif step>=round(1.25*expected):z[...,eos]=z[...,:eos].max(-1).values+20
    elif mode=="exact":
        if step<expected:z[...,eos]=-float("inf")
        else:z[...,eos]=z[...,:eos].max(-1).values+20
    else:raise ValueError(mode)
    return z

@torch.no_grad()
def fill(predict,x,steps=8):
    x=x.clone();count=int((x==1024).sum())
    for j in range(steps):
        m=x==1024
        if not m.any():break
        conf,ids=predict(x).softmax(-1).max(-1)
        left=math.floor(count*math.cos(math.pi*(j+1)/(2*steps))) if j+1<steps else 0
        k=max(1,int(m.sum())-left)
        ix=conf.masked_fill(~m,-1).topk(k,dim=-1).indices
        x.scatter_(1,ix,ids.gather(1,ix))
    assert x.min()>=0 and x.max()<1024
    return x

def low_conf_mask(x,confidence,ratio=.30):
    assert x.shape==confidence.shape and 0<ratio<=1
    k=max(1,round(x.size(1)*ratio))
    ix=confidence.topk(k,largest=False,dim=-1).indices
    return x.clone().scatter_(1,ix,1024)

def selftest():
    z=torch.zeros(1,1025)
    assert duration_logits(z,12,30,"band")[0,-1]==-float("inf")
    assert duration_logits(z,30,30,"exact").argmax()==1024
    assert duration_logits(z,20,30,"soft")[0,-1]<0
    x=torch.tensor([[1,2,3,4,5]])
    m=low_conf_mask(x,torch.tensor([[.9,.1,.8,.2,.7]]),.4)
    assert m.tolist()==[[1,1024,3,1024,5]]
    y=fill(lambda t:torch.nn.functional.one_hot(torch.full_like(t,7),1024).float()*10,m,8)
    assert y.tolist()==[[1,7,3,7,5]]
    print("Duration control, low-confidence mask and selective fill tests passed")
if __name__=="__main__":selftest()
