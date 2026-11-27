"""Research-only V5 CFM solvers and interval guidance (no weights changed).
16 midpoint steps = 32 velocity evaluations, matching Euler32.
"""
import torch

def integrate(x,field,steps,prompt_len,method):
    if steps<1 or method not in ("euler","midpoint"):raise ValueError("Invalid solver")
    d=1./steps
    for j in range(steps):
        k=field(x,j*d)
        if method=="midpoint":
            mid=x+(d/2)*k
            mid[...,:prompt_len]=0
            k=field(mid,(j+.5)*d)
        x=x+d*k
        x[...,:prompt_len]=0
    return x

@torch.inference_mode()
def research_v5(cfm,mu,x_lens,prompt,n_timesteps,temperature=.875,inference_cfg_rate=1.30,generator=None,
                *,method="midpoint",guidance_interval=(0.,1.),diagnostics=None,velocity_adapter=None):
    lo,hi=guidance_interval
    if not 0<=lo<=hi<=1 or inference_cfg_rate<0:raise ValueError("Invalid guidance schedule")
    if cfm.use_step_embedding or cfm.cfg_drop_text:raise ValueError("V5 only")
    B,T=mu.shape[:2];p=prompt.size(-1)
    if cfm.noise_temperature is not None:temperature=cfm.noise_temperature
    x=torch.randn((B,cfm.in_channels,T),device=mu.device,dtype=mu.dtype,generator=generator)*temperature
    px=torch.zeros_like(x);px[...,:p]=prompt;x[...,:p]=0
    mu=mu.transpose(2,1)
    static=cfm.estimator.prepare_static_cache(px,x_lens,mu) if cfm.use_conditioner_cache and cfm.use_static_cache else None
    cache=None
    counts={"conditional_evals":0,"negative_evals":0}
    def field(state,t):
        nonlocal cache
        ts=torch.full((B,),t,device=mu.device,dtype=mu.dtype)
        kwargs=dict(text0=mu,use_grad_ckpt=False,drop_text=False,infer=True,static_cache=static,dbcache=None)
        counts["conditional_evals"]+=1
        v,emb,_=cfm.estimator(state,px,x_lens,ts,drop_audio_cond=False,text_cache=cache,dbcache_branch=0,**kwargs)
        if cfm.use_conditioner_cache:cache=emb
        v=v.transpose(2,1)
        if inference_cfg_rate>1e-5 and lo<=t<hi:
            counts["negative_evals"]+=1
            neg,_,_=cfm.estimator(state,px,x_lens,ts,drop_audio_cond=True,text_cache=cache,dbcache_branch=1,**kwargs)
            v=v+(v-neg.transpose(2,1))*inference_cfg_rate
        if velocity_adapter is not None:
            v=v+velocity_adapter(state,v,mu,ts)
        return v
    result=integrate(x,field,int(n_timesteps),p,method)
    if diagnostics is not None:diagnostics.update(counts)
    return result


def midpoint_v5(cfm,mu,x_lens,prompt,n_timesteps,temperature=.875,inference_cfg_rate=1.30,generator=None):
    return research_v5(cfm,mu,x_lens,prompt,n_timesteps,temperature,inference_cfg_rate,generator,method="midpoint")

def selftest():
    x=torch.ones(1,1,5);x[...,:2]=0
    a=integrate(x.clone(),lambda z,t:z,32,2,"euler")
    b=integrate(x.clone(),lambda z,t:z,16,2,"midpoint")
    exact=torch.exp(torch.tensor(1.))
    assert (b[...,2:]-exact).abs().max()<(a[...,2:]-exact).abs().max()
    assert torch.count_nonzero(b[...,:2])==0
    class Estimator:
        def __call__(self,state,prompt,lengths,t,drop_audio_cond=False,**kwargs):
            return (torch.ones_like(state).transpose(2,1)*(0. if drop_audio_cond else 1.),None,None)
    from types import SimpleNamespace
    cfm=SimpleNamespace(use_step_embedding=False,cfg_drop_text=False,noise_temperature=.875,
        in_channels=2,use_conditioner_cache=False,use_static_cache=False,estimator=Estimator())
    mu=torch.zeros(1,5,3);length=torch.tensor([5]);prompt=torch.zeros(1,2,2)
    def run(interval,method,steps):
        d={}
        y=research_v5(cfm,mu,length,prompt,steps,generator=torch.Generator().manual_seed(9),
                      method=method,guidance_interval=interval,diagnostics=d)
        assert torch.count_nonzero(y[...,:2])==0
        return y,d
    full,f=run((0.,1.),"euler",32);limited,l=run((.25,.75),"euler",32)
    mid,m=run((0.,1.),"midpoint",16)
    assert f==m=={"conditional_evals":32,"negative_evals":32}
    assert l=={"conditional_evals":32,"negative_evals":16}
    torch.testing.assert_close(full,mid,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(full[...,2:]-limited[...,2:],torch.full_like(full[...,2:],.65),atol=1e-5,rtol=1e-5)
    print("Midpoint and guidance interval selftests passed")
if __name__=="__main__":selftest()
