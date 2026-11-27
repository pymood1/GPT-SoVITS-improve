"""Cross-model WAV-domain log-Mel distance, not MCD/MOS.
Same reference and target text required for synthetic-pair interpretation.
Different speakers or target utterances must not be treated as fidelity labels.
"""
import numpy as np
from scipy.signal import resample_poly
from scipy.spatial.distance import cdist
from math import gcd

SR=24000
NFFT=1024
HOP=240
BINS=80

def features(wave,sr,*,max_hz=10000,analysis_sr=24000):
    if analysis_sr not in (24000,32000):raise ValueError('Unsupported analysis rate')
    SR=analysis_sr;NFFT=1024 if SR==24000 else 1280;HOP=SR//100
    if not 1000<=max_hz<=min(SR/2,sr/2):
        raise ValueError('Requested bandwidth exceeds source Nyquist')
    x=np.asarray(wave,dtype=np.float64)
    if x.ndim==2:x=x.mean(axis=1)
    if x.ndim!=1 or len(x)<sr/10 or not np.isfinite(x).all():
        raise ValueError("Invalid waveform")
    g=gcd(int(sr),SR)
    if sr!=SR:x=resample_poly(x,SR//g,int(sr)//g)
    duration=len(x)/SR
    if duration>20:raise ValueError("Unexpectedly long diagnostic waveform")
    # Remove only leading/trailing low-energy frames, retaining interior pauses.
    n=HOP
    f=np.lib.stride_tricks.sliding_window_view(x,n)[::n]
    energy=np.mean(f*f,axis=1)
    peak=energy.max()
    if peak<1e-12:raise ValueError("Silent waveform")
    idx=np.flatnonzero(energy>=peak*1e-4)
    x=x[idx[0]*n:min(len(x),(idx[-1]+1)*n)]
    rms=np.sqrt(np.mean(x*x))
    x=x*(.05/rms)
    if len(x)<NFFT:x=np.pad(x,(0,NFFT-len(x)))
    frames=np.lib.stride_tricks.sliding_window_view(x,NFFT)[::HOP]
    power=np.abs(np.fft.rfft(frames*np.hanning(NFFT),axis=1))**2
    hz=np.fft.rfftfreq(NFFT,1/SR)
    mel=lambda z:2595*np.log10(1+z/700)
    edges=700*(10**(np.linspace(mel(80),mel(max_hz),BINS+2)/2595)-1)
    bank=np.maximum(0,np.minimum((hz[None,:]-edges[:-2,None])/(edges[1:-1,None]-edges[:-2,None]),
        (edges[2:,None]-hz[None,:])/(edges[2:,None]-edges[1:-1,None])))
    bank/=bank.sum(axis=1,keepdims=True).clip(1e-12)
    logmel=10*np.log10(np.maximum(power@bank.T,1e-10))
    return logmel,duration

def compare(a,b):
    x,dx=a;y,dy=b
    c=cdist(x,y,metric="cityblock")/x.shape[1]
    n,m=c.shape
    # DTW with a 25%-of-duration normalized diagonal band.
    dp=np.full((n+1,m+1),np.inf); steps=np.zeros((n+1,m+1),np.int32)
    dp[0,0]=0
    for i in range(1,n+1):
        lo=max(1,int((i/n-.25)*m));hi=min(m,int((i/n+.25)*m)+1)
        for j in range(lo,hi+1):
            opts=(dp[i-1,j-1],dp[i-1,j],dp[i,j-1])
            k=int(np.argmin(opts))
            pi,pj=((i-1,j-1),(i-1,j),(i,j-1))[k]
            dp[i,j]=opts[k]+c[i-1,j-1]
            steps[i,j]=steps[pi,pj]+1
    if not np.isfinite(dp[n,m]):raise ValueError("No alignment")
    return {"dtw_logmel_l1_db":float(dp[n,m]/steps[n,m]),
            "path_stretch":float(steps[n,m]/max(n,m)),
            "duration_ratio":float(dx/dy),
            "mean_band_l1_db":float(np.abs(x.mean(0)-y.mean(0)).mean())}

def selftest():
    t=np.arange(SR)/SR
    x=.1*np.sin(2*np.pi*440*t);y=.1*np.sin(2*np.pi*880*t)
    a=features(x,SR);b=features(x*.3,SR)
    assert compare(a,a)["dtw_logmel_l1_db"]==0
    assert compare(a,b)["dtw_logmel_l1_db"]<1e-8
    assert compare(a,features(y,SR))["dtw_logmel_l1_db"]>1
    assert compare(features(x,SR,max_hz=7600),features(x*.3,SR,max_hz=7600))["dtw_logmel_l1_db"]<1e-8
    print("Spectral compare selftest passed")

if __name__=="__main__":selftest()
