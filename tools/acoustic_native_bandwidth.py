"""Native 48 kHz waveform bandwidth audit; never resample source.
Band energy is descriptive; cannot establish original recording provenance.
"""
import argparse,json
from pathlib import Path
import numpy as np

BANDS=((80,4000),(4000,8000),(8000,12000),(12000,16000),(16000,20000),(20000,24000))
def audit(x,sr):
    x=np.asarray(x,dtype=np.float64)
    if sr!=48000 or x.ndim!=1 or len(x)<sr or not np.isfinite(x).all():
        raise ValueError("Require native 48kHz mono waveform >=1 second")
    n=2048;hop=1024
    frames=np.lib.stride_tricks.sliding_window_view(x,n)[::hop]
    p=np.mean(frames**2,axis=1)
    anchor=np.quantile(p,.9)
    if anchor<1e-12:raise ValueError("Silent or insufficient signal")
    active=p>=anchor*.1
    q=np.abs(np.fft.rfft(frames[active]*np.hanning(n),axis=1))**2
    hz=np.fft.rfftfreq(n,1/sr)
    total=q[:,hz>=80].sum()
    if total<=0:raise ValueError("No speech-band energy")
    bands={f"{lo}-{hi}":float(10*np.log10(max(q[:,(hz>=lo)&(hz<hi)].sum()/total,1e-15))) for lo,hi in BANDS}
    return dict(sample_rate=sr,seconds=len(x)/sr,active_frames=int(active.sum()),
        rms_dbfs=float(20*np.log10(max(np.sqrt(np.mean(x*x)),1e-15))),
        clipped_fraction=float(np.mean(np.abs(x)>=.999)),band_relative_db=bands,
        caveat="Low high-band energy is a review flag, not proof of upsampling; recording provenance required.")
def selftest():
    t=np.arange(48000)/48000
    low=.1*np.sin(2*np.pi*440*t)
    high=low+.02*np.sin(2*np.pi*18000*t)
    a,b=audit(low,48000),audit(high,48000)
    assert b["band_relative_db"]["16000-20000"]>a["band_relative_db"]["16000-20000"]+30
    assert abs(audit(high*.25,48000)["band_relative_db"]["16000-20000"]-b["band_relative_db"]["16000-20000"])<1e-8
    try:audit(low,16000)
    except ValueError:pass
    else:raise AssertionError("16kHz accepted")
    print("Native bandwidth audit selftest passed")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("wav",nargs="?",type=Path);p.add_argument("--selftest",action="store_true")
    a=p.parse_args()
    if a.selftest:selftest()
    else:
        if a.wav is None:p.error("WAV required")
        import soundfile as sf
        x,sr=sf.read(a.wav,dtype="float64")
        print(json.dumps(audit(x,sr),indent=2,allow_nan=False))
