"""Evaluate teacher, compact round-11 stack and full-width exported SwiGLU on held-out speech.
Same sampler, same full-forward decoder mode, identical S2 and acoustic seed.
No production changes, no raw text in aggregate reports.
"""
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


class StackGenerator:
    def __init__(self,teacher,path):
        self.teacher=teacher
        saved=torch.load(path,map_location="cpu",weights_only=True)
        assert saved["format"]=="s1-progressive-stack-v1"
        self.cfg=SimpleNamespace(codes=teacher.EOS)
        assert saved["depth"]==len(teacher.h.layers)
        assert saved["dim"]==teacher.h.layers[0].linear1.in_features
        device=next(teacher.parameters()).device
        self.mods={}
        for i,state in saved["replacements"].items():
            m=SwiFFN(saved["dim"],saved["width"]).to(device)
            m.load_state_dict(state,strict=True)
            self.mods[int(i)]=m.eval().requires_grad_(False)
        assert self.mods and all(0<=i<saved["depth"] for i in self.mods)

    @torch.inference_mode()
    def generate(self,x,b,prompt,*,max_new,min_new,top_k,repetition_penalty,generator):
        y=prompt
        for i in range(max_new):
            z,_=logits(self.teacher,{"x":x,"b":b,"y":y},self.mods,x.device)
            token=sample_next(z[:,-1],y,eos=self.teacher.EOS,allow_eos=i>=min_new,
                              top_k=top_k,repetition_penalty=repetition_penalty,generator=generator)
            if token.item()==self.teacher.EOS:
                break
            y=torch.cat((y,token),dim=1)
        return y[:,prompt.size(1):],None


def main():
    p=argparse.ArgumentParser()
    for name in ("teacher","student","prior-student","s2","manifest","output"):
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
        student=StackGenerator(teacher,a.student)
        prior=StackGenerator(teacher,a.prior_student)
        for ri,ref in enumerate(m["references"]):
            if ri%a.worker_count!=a.worker_index:
                continue
            engine.set_ref_audio(ref["path"])
            cache={}
            for ti,target in enumerate(m["targets"]):
                seed=2027+ri*1009+ti
                for arm in ("teacher","prior_student","student"):
                    row={"arm":arm,"reference_index":ri+1,"text_index":ti+1,"seed":seed}
                    try:
                        sr,wave,stats=synthesize(engine,student,teacher,ref,target,seed,a.max_tokens,arm,
                                                 cache=cache,prior_student=prior)
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
