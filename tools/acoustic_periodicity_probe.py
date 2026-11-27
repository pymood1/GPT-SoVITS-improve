"""Voiced-periodicity diagnostic, not clinical HNR/CPP, MOS or noise SNR."""
import numpy as np
from scipy.signal import resample_poly
from math import gcd

def probe(x,sr):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=1 or not np.isfinite(x).all() or len(x)<sr*.1:raise ValueError("Invalid audio")
    g=gcd(sr,16000)
    x=resample_poly(x,16000//g,sr//g) if sr!=16000 else x
    n=960;hop=320
    f=np.lib.stride_tricks.sliding_window_view(x,n)[::hop].copy()
    f-=f.mean(1,keepdims=True)
    power=np.mean(f*f,1);active=power>=max(np.quantile(power,.9)*.1,1e-12)
    f=f[active]
    if not len(f):return {"active_frames":0,"periodic_frames":0,"periodic_fraction":0.,"median_periodicity":None,"median_residual_db":None}
    # Compare overlapping lagged samples without window attenuation.
    lags=np.arange(32,267)
    ac=[]
    for lag in lags:
        a=f[:,:-lag];b=f[:,lag:]
        ac.append(np.sum(a*b,1)/np.sqrt(np.maximum(np.sum(a*a,1)*np.sum(b*b,1),1e-24)))
    r=np.array(ac).T
    # Use maximum local peak, not endpoint; confidence gate independent of energy.
    peaks=(r[:,1:-1]>=r[:,:-2])&(r[:,1:-1]>=r[:,2:])
    best=np.max(np.where(peaks,r[:,1:-1],-1.),axis=1)
    valid=best>=.5
    z=np.clip(best[valid],1e-6,1-1e-6)
    return {"active_frames":len(f),"periodic_frames":int(valid.sum()),
            "periodic_fraction":float(valid.mean()),
            "median_periodicity":float(np.median(z)) if len(z) else None,
            "median_residual_db":float(np.median(10*np.log10((1-z)/z))) if len(z) else None}

def selftest():
    t=np.arange(16000)/16000
    x=np.sin(2*np.pi*200*t)
    y=x+np.random.default_rng(13).normal(0,.3,len(t))
    a,b=probe(x,16000),probe(y,16000)
    assert a["median_periodicity"]>b["median_periodicity"]
    assert abs(probe(.1*x,16000)["median_periodicity"]-a["median_periodicity"])<1e-8
    print("Voiced-periodicity diagnostic selftest passed")
if __name__=="__main__":selftest()
