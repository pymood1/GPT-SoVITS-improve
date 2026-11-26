"""Round 15: all four arms use the same cached sampler and frozen FP32 V5dev."""
import argparse
import json
import sys
import tempfile
from pathlib import Path
import torch
from types import SimpleNamespace
from AR.models.s1_next import sample_next
from tools.train_s1_progressive_ffn import SwiFFN
from tools.train_s1_progressive_stack import logits
from tools.evaluate_s1_next_waveforms import read_manifest, synthesize, summarize
from tools.train_s1_next import teacher_logits


from tools.train_s1_rms_polish import generators

def main():
    p=argparse.ArgumentParser()
    for name in ("teacher","stack","resume","refined","s2","manifest","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--worker-index",type=int,required=True)
    p.add_argument("--worker-count",type=int,default=2)
    p.add_argument("--max-tokens",type=int,default=384)
    a=p.parse_args()
    from TTS_infer_pack.TTS import TTS,TTS_Config
    import soundfile as sf
    m=read_manifest(a.manifest.resolve())
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    with tempfile.TemporaryDirectory(prefix="s1-stack-waveform-") as tmp:
        cfg=TTS_Config({"custom":dict(TTS_Config.default_configs["v5dev"],
                       device="cuda:0",is_half=False,t2s_weights_path=str(a.teacher.resolve()),
                       vits_weights_path=str(a.s2.resolve()))})
        cfg.configs_path=str(Path(tmp)/"tts.yaml")
        engine=TTS(cfg)
        teacher=engine.t2s_model.model.eval()
        gs=generators(teacher,a.stack,a.resume,a.refined,next(teacher.parameters()).device)
        for ri,ref in enumerate(m["references"]):
            if ri%a.worker_count!=a.worker_index:
                continue
            engine.set_ref_audio(ref["path"])
            for ti,target in enumerate(m["targets"]):
                seed=2035+ri*1009+ti
                for arm in ("teacher","round12","round14","student"):
                    row={"arm":arm,"reference_index":ri+1,"text_index":ti+1,"seed":seed}
                    try:
                        sr,wave,stats=synthesize(engine,gs[arm],teacher,ref,target,seed,a.max_tokens,"student",
                                                 cache=None)
                        name=f"{arm}-{ri+1:02}-{ti+1:02}.wav"
                        sf.write(a.output/name,wave,sr,subtype="PCM_16")
                        row.update(status="ok",file=name,**stats)
                    except Exception as exc:
                        import traceback
                        with (a.output/"private-errors.log").open("a") as f:
                            f.write(traceback.format_exc()+"\n")
                        row.update(status="failed",error_type=type(exc).__name__)
                    rows.append(row)
                    print(json.dumps({k:v for k,v in row.items() if k!="file"},allow_nan=False),flush=True)
                    (a.output/"waveform-metrics.json").write_text(json.dumps({"rows":rows,"summary":summarize(rows)},allow_nan=False))
    scoring={"texts":m["targets"],"references":[r["path"] for r in m["references"]],
             "samples":[{"label":r["arm"],"path":str((a.output/r["file"]).resolve()),
                         "reference_index":r["reference_index"],"text_index":r["text_index"]}
                        for r in rows if r["status"]=="ok"],"cpu_threads":2}
    (a.output/"scoring-input.json").write_text(json.dumps(scoring,ensure_ascii=False))
    if not scoring["samples"]:
        raise RuntimeError("No valid synthesis")


if __name__=="__main__":
    main()
