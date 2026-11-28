"""Research-only masked refinement with held-out token confidence.
Unlike scoring an already visible token, each scored token is hidden from
the model. No target audio, true length or ASR labels enter decoding.
"""
import math
import torch

MASK=1024

@torch.no_grad()
def heldout_confidence(predict,tokens,groups=4):
    if tokens.ndim!=2 or tokens.size(0)!=1 or tokens.size(1)<1 or not 1<=groups:
        raise ValueError("Expected nonempty single sequence and positive groups")
    if tokens.min()<0 or tokens.max()>=MASK:raise ValueError("Invalid semantic IDs")
    n=tokens.size(1);groups=min(groups,n)
    scores=torch.empty_like(tokens,dtype=torch.float32)
    pos=torch.arange(n,device=tokens.device)
    for g in range(groups):
        ix=pos%groups==g
        hidden=tokens.clone();hidden[:,ix]=MASK
        logp=predict(hidden).float().log_softmax(-1)
        assert logp.shape==(1,n,MASK)
        scores[:,ix]=logp[:,ix].gather(-1,tokens[:,ix,None]).squeeze(-1)
    assert torch.isfinite(scores).all()
    return scores,groups

@torch.no_grad()
def refine(predict,tokens,rounds=3,groups=4,ratio=.25,minimum=.05,locked=None):
    if not 1<=rounds or not 0<minimum<=ratio<=1:raise ValueError("Invalid schedule")
    x=tokens.clone()
    if locked is None:locked=torch.zeros_like(x,dtype=torch.bool)
    if locked.shape!=x.shape or locked.dtype!=torch.bool:raise ValueError("Invalid anchors")
    free=int((~locked).sum());trace=[];calls=0
    for r in range(rounds):
        if not free:break
        conf,c=heldout_confidence(predict,x,groups);calls+=c
        fraction=minimum+(ratio-minimum)*math.cos(math.pi*r/(2*rounds))
        k=max(1,min(free,round(free*fraction)))
        ix=conf.masked_fill(locked,float("inf")).topk(k,largest=False,dim=-1).indices
        hidden=x.clone().scatter_(1,ix,MASK)
        logits=predict(hidden);calls+=1
        assert logits.shape==(1,x.size(1),MASK)
        ids=logits.argmax(-1).gather(1,ix)
        changed=int((ids!=x.gather(1,ix)).sum())
        x.scatter_(1,ix,ids)
        trace.append(dict(round=r+1,remasked=k,changed=changed,
            mean_heldout_logp=float(conf.mean()),predict_calls=calls))
        if not changed:break
    assert torch.equal(x[locked],tokens[locked])
    return x,dict(predict_calls=calls,rounds=trace)

def selftest():
    x=torch.tensor([[7,7,3,7,7,4,7,7]])
    def copy_trap(z):
        # Visible tokens trivially copy. Hidden positions predict clean id7.
        ids=torch.where(z==MASK,7,z)
        return torch.nn.functional.one_hot(ids,MASK).float()*12
    conf,c=heldout_confidence(copy_trap,x,4)
    assert c==4 and conf[0,2]<conf[0,0] and conf[0,5]<conf[0,0]
    y,trace=refine(copy_trap,x,rounds=2,ratio=.25)
    assert y.eq(7).all() and trace["rounds"][0]["changed"]==2
    locked=torch.zeros_like(x,dtype=torch.bool);locked[:,2]=True
    y,_=refine(copy_trap,x,rounds=2,ratio=.25,locked=locked)
    assert y[0,2]==3 and y[0,5]==7
    y,trace=refine(copy_trap,x,locked=torch.ones_like(locked))
    assert torch.equal(y,x) and trace["predict_calls"]==0
    print("Held-out confidence, copy-trap detection, selective repair and locked-token tests passed")
if __name__=="__main__":selftest()
