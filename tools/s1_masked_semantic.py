"""v22 masked semantic generator; research-only, not a pretrained MaskGCT implementation."""
import math
import torch
from torch import nn
import torch.nn.functional as F

MASK=1024
class MaskedSemantic(nn.Module):
    def __init__(self,phone_vocab,dim=256,layers=4):
        super().__init__()
        self.phone=nn.Embedding(phone_vocab,dim)
        self.unit=nn.Embedding(1025,dim)
        self.bert=nn.Linear(768,dim,bias=False)
        nn.init.zeros_(self.bert.weight)
        self.pos=nn.Embedding(1024,dim)
        self.segment=nn.Embedding(3,dim)
        self.duration=nn.Sequential(nn.Linear(dim*2,dim),nn.SiLU(),nn.Linear(dim,1))
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,4,dim*4,dropout=.1,batch_first=True,norm_first=True),layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(dim)
        self.out=nn.Linear(dim,1024)
    def conditions(self,r):
        # Only reference tokens and target text enter the condition, never target audio.
        phone=self.phone(r["target"]["x"])+self.bert(r["target"]["bb"]["word"].transpose(1,2)[...,:768])
        prompt=self.unit(r["prompt"])
        pooled=torch.cat((phone.mean(1),prompt.mean(1)),1)
        # Predict log semantic frames per phone, rather than an EOS.
        loglen=self.duration(pooled).squeeze(-1)+math.log(r["target"]["x"].size(1))
        return phone,prompt,loglen
    def forward(self,r,masked):
        phone,prompt,loglen=self.conditions(r)
        target=self.unit(masked)
        seq=[]
        for i,z in enumerate((phone,prompt,target)):
            p=torch.arange(z.size(1),device=z.device)
            assert z.size(1)<=1024
            seq.append(z+self.pos(p)[None]+self.segment.weight[i][None,None])
        h=self.encoder(torch.cat(seq,1))
        return self.out(self.norm(h[:,-masked.size(1):])),loglen
    @torch.no_grad()
    def length(self,r):
        return int(self.conditions(r)[2].exp().round().clamp(11,512).item())

def mask_for(y,ratio,g):
    n=y.size(1);k=max(1,min(n,round(n*ratio)))
    ix=torch.randperm(n,device=y.device,generator=g)[:k]
    m=torch.zeros_like(y,dtype=torch.bool);m[:,ix]=True
    return torch.where(m,MASK,y),m

@torch.no_grad()
def decode(model,r,n,steps,seed,initial=None,repair_ratio=.0):
    assert 1<=n<=512 and steps>=1
    device=r["prompt"].device
    g=torch.Generator(device=device).manual_seed(seed)
    if initial is None:
        x=torch.full((1,n),MASK,device=device,dtype=torch.long)
    else:
        assert initial.shape==(1,n) and initial.max()<MASK
        x,_=mask_for(initial,repair_ratio,g)
    count=int((x==MASK).sum())
    for step in range(steps):
        masked=x==MASK
        if not masked.any():break
        lg,_=model(r,x)
        p=lg.softmax(-1);confidence,ids=p.max(-1)
        left=math.floor(count*math.cos(math.pi*(step+1)/(2*steps))) if step+1<steps else 0
        keep=max(1,int(masked.sum())-left)
        chosen=confidence.masked_fill(~masked,-1).topk(keep,dim=-1).indices
        x.scatter_(1,chosen,ids.gather(1,chosen))
    assert x.min()>=0 and x.max()<MASK
    return x

def selftest():
    torch.manual_seed(3);m=MaskedSemantic(40,dim=32,layers=1)
    r={"target":{"x":torch.tensor([[1,2,3,4]]),"bb":{"word":torch.randn(1,1024,4)}},"prompt":torch.tensor([[1,4,8]])}
    y=torch.randint(1024,(1,13));g=torch.Generator().manual_seed(1)
    z,mask=mask_for(y,.5,g);lg,ll=m(r,z)
    assert lg.shape==(1,13,1024) and mask.sum()==6
    loss=F.cross_entropy(lg[mask],y[mask])+F.smooth_l1_loss(ll,torch.tensor([math.log(13.)]))
    loss.backward()
    assert m.bert.weight.grad.abs().sum()>0
    m.eval()
    x=decode(m,r,13,4,1);assert x.shape==y.shape
    x=decode(m,r,13,4,1,initial=y,repair_ratio=.3)
    assert (x==y).sum()>=9
    print("Masked semantic forward, duration loss, gradient, iterative decode and retained-token tests passed")
if __name__=="__main__":selftest()
