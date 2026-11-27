"""Research-only explicit midpoint integrator for V5 CFM (no weights changed).
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
def midpoint_v5(cfm,mu,x_lens,prompt,n_timesteps,temperature=.875,inference_cfg_rate=1.30,generator=None):
    if cfm.use_step_embedding or cfm.cfg_drop_text:raise ValueError("V5 only")
    B,T=mu.shape[:2];p=prompt.size(-1)
    if cfm.noise_temperature is not None:temperature=cfm.noise_temperature
    x=torch.randn((B,cfm.in_channels,T),device=mu.device,dtype=mu.dtype,generator=generator)*temperature
    px=torch.zeros_like(x);px[...,:p]=prompt;x[...,:p]=0
    mu=mu.transpose(2,1)
    static=cfm.estimator.prepare_static_cache(px,x_lens,mu) if cfm.use_conditioner_cache and cfm.use_static_cache else None
    cache=None
    def field(state,t):
        nonlocal cache
        ts=torch.full((B,),t,device=mu.device,dtype=mu.dtype)
        kwargs=dict(text0=mu,use_grad_ckpt=False,drop_text=False,infer=True,static_cache=static,dbcache=None)
        v,emb,_=cfm.estimator(state,px,x_lens,ts,drop_audio_cond=False,text_cache=cache,dbcache_branch=0,**kwargs)
        if cfm.use_conditioner_cache:cache=emb
        v=v.transpose(2,1)
        if inference_cfg_rate>1e-5:
            neg,_,_=cfm.estimator(state,px,x_lens,ts,drop_audio_cond=True,text_cache=cache,dbcache_branch=1,**kwargs)
            v=v+(v-neg.transpose(2,1))*inference_cfg_rate
        return v
    return integrate(x,field,int(n_timesteps),p,"midpoint")

def selftest():
    x=torch.ones(1,1,5);x[...,:2]=0
    a=integrate(x.clone(),lambda z,t:z,32,2,"euler")
    b=integrate(x.clone(),lambda z,t:z,16,2,"midpoint")
    exact=torch.exp(torch.tensor(1.))
    assert (b[...,2:]-exact).abs().max()<(a[...,2:]-exact).abs().max()
    assert torch.count_nonzero(b[...,:2])==0
    print("Midpoint integration selftest passed")
if __name__=="__main__":selftest()
