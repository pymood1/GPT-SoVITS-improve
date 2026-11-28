"""Fixed-mel factorial: reference gain x hard/soft tail compression. No CFM or S1 generation."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from tools.s1_s2_acoustic_isolation import stats,level_match

def mel_stats(m):
    a=np.asarray(m,dtype=np.float32)
    if not a.size or not np.isfinite(a).all():
        raise ValueError("Invalid mel")
    return {"mel_min":float(a.min()),"mel_max":float(a.max()),
            "mel_p001":float(np.quantile(a,.001)),"mel_p01":float(np.quantile(a,.01)),
            "mel_below_minus12":float(np.mean(a< -12)),
            "mel_above_2":float(np.mean(a>2))}

def soft_tail(m, low=-12., high=2., scale=2.):
    # Identity inside calibration range; continuous first derivative at both bounds.
    return torch.where(m < low, low + scale*torch.tanh((m-low)/scale),
                       torch.where(m > high, high + scale*torch.tanh((m-high)/scale), m))

def selftest():
    x=torch.tensor([-49.,-12.,-11.,0.,2.,20.])
    y=soft_tail(x)
    assert torch.equal(x[1:5],y[1:5])
    assert -14 <= y[0] < -12 and 2 < y[-1] <= 4
    assert torch.isfinite(y).all()
    from tools.acoustic_noise_probe import selftest as noise_test
    noise_test()
    m=np.array([-49.,-12.,-11.,0.,3.],np.float32)
    r=mel_stats(m)
    assert abs(r["mel_below_minus12"]-.2)<1e-6 and abs(r["mel_above_2"]-.2)<1e-6
    assert np.clip(m,-12,2).tolist()==[-12.,-12.,-11.,0.,2.]
    print("Acoustic mel range diagnostics passed")

@torch.inference_mode()
def main(a):
    from TTS_infer_pack.TTS import TTS,TTS_Config,mel_fn_v4,norm_spec,denorm_spec,resample
    from tools.s1_waveform_frontend import ensure_worker_nltk
    from tools.evaluate_s1_next_waveforms import features
    from module.models_v5 import synthesize_v5_mel
    ri=a.worker_index
    source=a.source/f"ref{ri+1:02}"
    texts=json.loads(a.texts.read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        cfg=TTS_Config({"custom":dict(TTS_Config.default_configs["v5dev"],device="cuda:0",
            is_half=False,t2s_weights_path=str(a.teacher.resolve()),vits_weights_path=str(a.s2.resolve()))})
        cfg.configs_path=str(Path(tmp)/"tts.yaml")
        engine=TTS(cfg)
        ensure_worker_nltk()
        assert engine.precision==torch.float32 and engine.configs.version=="v5dev"
        device=engine.configs.device
        vits=engine.vits_model.eval()
        sr=engine.vocoder_configs["sr"]
        assert sr==48000
        def save(name,mel,**extra):
            arr=mel.detach().float().cpu().numpy()
            wave=engine.vocoder(mel.float())[0,0].float().cpu().numpy()
            row={"arm":name,"reference_index":ri+1,"text_index":1,"status":"ok",
                 **stats(wave,sr),**mel_stats(arr),**extra}
            sf.write(a.output/(name+".wav"),np.clip(wave,-1,1),sr,subtype="PCM_16")
            leveled,gain=level_match(wave,max_gain=32.)
            sf.write(a.output/(name+"-level.wav"),leveled,sr,subtype="PCM_16")
            row["listening_gain"]=gain
            np.savez_compressed(a.output/(name+"-mel.npz"),mel=arr)
            rows.append(row)
            (a.output/"metrics.json").write_text(json.dumps(rows,indent=2,allow_nan=False))
            print(json.dumps(row,allow_nan=False),flush=True)

        # Both source mels were synthesized with the same semantic tokens, seed and CFG 1.30.
        from tools.acoustic_noise_probe import probe
        for label, prior_name in (("raw","teacher-baseline"),("gain","teacher-ref-gain")):
            mel=torch.from_numpy(np.load(source/(prior_name+"-mel.npz"))["mel"]).to(device)
            for transform, transformed in (("identity",mel),("hard",mel.clamp(-12,2)),("soft",soft_tail(mel))):
                name=label+"-"+transform
                save(name,transformed)
                if transform=="identity":
                    prior, rate=sf.read(source/(prior_name+".wav"),dtype="float32")
                    now, now_rate=sf.read(a.output/(name+".wav"),dtype="float32")
                    assert prior.shape==now.shape and rate==now_rate==sr
                    delta=float(np.max(np.abs(prior-now)))
                    rows[-1]["saved_pcm_max_delta"]=delta
                    assert delta<=1/32768, "Saved mel replay mismatch"
                w,rate=sf.read(a.output/(name+"-level.wav"),dtype="float64")
                rows[-1]["noise_proxies"]=probe(w,rate)
                (a.output/"metrics.json").write_text(json.dumps(rows,indent=2,allow_nan=False))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    for name in ("teacher","s2","source","texts","output"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--worker-index",type=int,default=0)
    a=p.parse_args()
    if a.selftest:selftest()
    else:main(a)
