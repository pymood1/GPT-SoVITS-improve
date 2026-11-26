"""CPU-only comparative waveform diagnostics; not a perceptual noise estimator.
Usage: python acoustic_noise_probe.py /path/to/acoustic-cfg --output report.json
Run --selftest for synthetic gain-invariance and noise sensitivity checks.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def probe(wave, sr):
    x = np.asarray(wave, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or sr < 16000 or not np.isfinite(x).all():
        raise ValueError('Invalid audio or sample rate')
    n, hop = round(sr * .04), round(sr * .02)
    if x.size < n or np.max(np.abs(x)) < 1e-10:
        raise ValueError('Audio too short or silent')
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    p = np.mean(frames ** 2, axis=1)
    anchor = np.quantile(p, .9)
    if anchor < 1e-20:
        raise ValueError('Insufficient active signal')
    rel = p / anchor
    quiet = rel <= .01
    active = rel >= .1
    # Common fixed 4–8 kHz band permits comparison at both 32 and 48 kHz.
    power = np.abs(np.fft.rfft(frames * np.hanning(n), axis=1)) ** 2
    hz = np.fft.rfftfreq(n, 1 / sr)
    band = power[:, (hz >= 4000) & (hz < 8000)].sum(axis=1)
    total = power[:, (hz >= 80) & (hz < 8000)].sum(axis=1)
    def ratio(mask):
        if not np.any(mask) or total[mask].sum() <= 0:
            return None
        return float(band[mask].sum() / total[mask].sum())
    return dict(seconds=x.size / sr, sample_rate=sr, frames=len(p),
                quiet_frames=int(quiet.sum()), active_frames=int(active.sum()),
                quiet_p10_relative_db=float(10*np.log10(max(np.quantile(rel,.1),1e-15))),
                quiet_hf_ratio=ratio(quiet), active_hf_ratio=ratio(active),
                clipped_fraction=float(np.mean(np.abs(x)>=.999)))


def selftest():
    sr=48000
    rng=np.random.default_rng(52)
    t=np.arange(sr*2)/sr
    clean=.15*np.sin(2*np.pi*440*t)*(t<1)
    noisy=clean+rng.normal(0,.003,t.shape)
    a,b=probe(noisy,sr),probe(noisy*.2,sr)
    for k in ('quiet_p10_relative_db','quiet_hf_ratio','active_hf_ratio'):
        assert abs(a[k]-b[k])<1e-9, k
    assert a['quiet_p10_relative_db']>probe(clean,sr)['quiet_p10_relative_db']+20
    assert a['quiet_hf_ratio']>.3
    assert a['quiet_frames']>0 and a['active_frames']>0
    for bad in (np.zeros(sr), np.array([np.nan]), np.ones(10)):
        try:
            probe(bad,sr)
        except ValueError:
            continue
        raise AssertionError('Invalid input accepted')
    print('Acoustic probe selftest passed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',nargs='?',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--selftest',action='store_true')
    a=p.parse_args()
    if a.selftest:
        selftest()
        return
    if a.root is None or not a.root.is_dir() or a.output is None:
        p.error('root directory and --output are required')
    rows=[]
    for path in sorted(a.root.glob('ref*/*-level.wav')):
        wave,sr=sf.read(path,dtype='float64')
        rows.append(dict(reference=path.parent.name,arm=path.stem[:-6],**probe(wave,sr)))
    if not rows:
        raise ValueError('No level-matched WAVs found')
    a.output.write_text(json.dumps(dict(rows=rows, caveat='Descriptive proxies; not SNR or MOS. Pauses and phonetic content confound comparisons.'),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
