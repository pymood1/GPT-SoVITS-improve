"""Controlled, deterministic low-quality inputs from native 48kHz originals."""
import numpy as np
from scipy.signal import resample_poly,fftconvolve

def degrade(wave,sr,kind,seed):
    x=np.asarray(wave,dtype=np.float64)
    if sr!=48000 or x.ndim!=1 or len(x)<sr or not np.isfinite(x).all():raise ValueError("Native mono 48kHz required")
    rng=np.random.default_rng(seed)
    if kind=="clean":return x.astype(np.float32)
    if kind=="noise10":
        n=rng.normal(size=len(x))
        y=x+n*np.sqrt(np.mean(x*x)/max(np.mean(n*n),1e-12)/10.)
    elif kind=="band8":
        y=resample_poly(resample_poly(x,1,6),6,1)[:len(x)]
    elif kind=="room":
        ir=np.zeros(int(sr*.18));ir[0]=1
        for delay,gain in ((.017,.35),(.043,.25),(.087,.12),(.141,.06)):ir[int(delay*sr)]=gain
        y=fftconvolve(x,ir)[:len(x)]
    else:raise ValueError("Unknown corruption")
    # Keep timing and prevent clipping; preserve original RMS where possible.
    y*=min(np.sqrt(np.mean(x*x)/max(np.mean(y*y),1e-12)),.95/max(np.max(np.abs(y)),1e-12))
    assert y.shape==x.shape and np.isfinite(y).all()
    return y.astype(np.float32)

def selftest():
    t=np.arange(48000)/48000;x=.1*np.sin(2*np.pi*1000*t)+.02*np.sin(2*np.pi*10000*t)
    n=degrade(x,48000,"noise10",1)
    assert np.array_equal(n,degrade(x,48000,"noise10",1))
    assert not np.array_equal(n,degrade(x,48000,"noise10",2))
    y=degrade(x,48000,"band8",2)
    assert abs(np.fft.rfft(y)[10000])<abs(np.fft.rfft(x)[10000])*.01
    for k in ("noise10","band8","room"):
        assert len(degrade(x,48000,k,1))==len(x)
    print("Controlled degradation selftest passed")
if __name__=="__main__":selftest()
