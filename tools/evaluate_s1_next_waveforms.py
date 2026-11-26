"""Independent-reference waveform evaluation for S1 Next.

No production changes. Use a public, consented or otherwise authorized reference
manifest; output audio and transcripts remain local/private. Matched sampler,
frozen V5dev, reference and target text; never use target audio as prompt.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"GPT_SoVITS")]
os.environ.setdefault("version","v2")
os.environ.setdefault("hz","25hz")
os.environ.setdefault("language","ko_KR")

import torch
from AR.models.s1_next import NextConfig,S1Next
from tools.train_s1_next import teacher_generate


def normalized(text):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKC",text).casefold() if c.isalnum())


def read_manifest(path):
    m=json.loads(path.read_text(encoding="utf-8"))
    refs=m["references"]
    texts=m["targets"]
    if not 2<=len(refs)<=6 or not 2<=len(texts)<=12:
        raise ValueError("Use 2-6 references and 2-12 target texts")
    seen=set()
    ref_texts=set()
    for row in refs:
        p=(path.parent/row["path"]).resolve()
        if not p.is_file() or not normalized(row["text"]):
            raise ValueError("Invalid reference")
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError("Duplicate reference audio")
        seen.add(digest)
        ref_texts.add(normalized(row["text"]))
        row["path"]=str(p)
    keys=[normalized(t) for t in texts]
    if any(not k or k in ref_texts for k in keys) or len(set(keys))!=len(keys):
        raise ValueError("Reference and target transcripts must be distinct")
    return m


def features(engine,text):
    if text[-1] not in ".!?。！？":
        text+="."
    phones,bert,_=engine.text_preprocessor.segment_and_extract_feature_for_text(
        text,"all_ko",engine.configs.version)
    x=torch.as_tensor(phones,device=engine.configs.device,dtype=torch.long)[None]
    b=torch.as_tensor(bert,device=x.device,dtype=torch.float32)[None]
    if x.numel()==0 or b.shape!=(1,1024,x.size(1)):
        raise ValueError("Invalid frontend features")
    return x,b


def token_stats(ids,cap):
    n=ids.size(-1)
    prev=None
    longest=run=0
    for v in ids[0].tolist():
        run=run+1 if v==prev else 1
        prev=v
        longest=max(run,longest)
    return {"tokens":n,"hit_cap":n==cap,"longest_repeat":longest}


@torch.inference_mode()
def synthesize(engine,student,teacher,ref,target,seed,cap,arm):
    import numpy as np
    px,pb=features(engine,ref["text"])
    tx,tb=features(engine,target)
    x=torch.cat((px,tx),dim=1)
    b=torch.cat((pb,tb),dim=2)
    engine.prompt_cache["phones"]=px[0].tolist()
    prompt=engine.prompt_cache["prompt_semantic"].reshape(1,-1).long()
    started=time.monotonic()
    if arm=="teacher":
        ids=teacher_generate(teacher,x,b,prompt,cap=cap,top_k=15,penalty=1.35,seed=seed)
    else:
        g=torch.Generator(device=x.device).manual_seed(seed)
        ids,_=student.generate(x,b,prompt,max_new=cap,min_new=10,top_k=15,
                               repetition_penalty=1.35,generator=g)
    stats=token_stats(ids,cap)
    stats["s1_seconds"]=time.monotonic()-started
    if ids.numel()==0 or int(ids.max())>=student.cfg.codes:
        raise ValueError("Empty or invalid generated semantic tokens")
    # Equal acoustic sampling seed, identical frozen S2 and vocoder settings.
    torch.manual_seed(seed+100000)
    wave=engine.using_vocoder_synthesis(ids[:,None,:],tx,sample_steps=32,cfg_rate=1.30)
    wave=wave.detach().float().cpu().numpy()
    sr=int(engine.vocoder_configs["sr"])
    if sr!=48000 or wave.ndim!=1 or wave.size<sr//10 or not np.isfinite(wave).all():
        raise ValueError("Invalid waveform")
    rms=float(np.sqrt(np.mean(wave.astype(np.float64)**2)))
    if rms<1e-5:
        raise ValueError("Silent waveform")
    stats.update(duration=wave.size/sr,rms=rms,total_seconds=time.monotonic()-started,
                 clipped_fraction=float(np.mean(np.abs(wave)>1.)))
    return sr,np.clip(wave,-1.,1.),stats


def summarize(rows):
    out={}
    for arm in ("teacher","student"):
        subset=[r for r in rows if r["arm"]==arm]
        successful=[r for r in subset if r["status"]=="ok"]
        out[arm]={"attempts":len(subset),"successes":len(successful)}
        if successful:
            for k in ("tokens","duration","s1_seconds","total_seconds","longest_repeat","hit_cap"):
                out[arm]["mean_"+k]=sum(float(r[k]) for r in successful)/len(successful)
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--student",type=Path,required=True)
    p.add_argument("--s2",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-tokens",type=int,default=512)
    args=p.parse_args()
    if not 32<=args.max_tokens<=1024:
        p.error("Use a predeclared generation cap of 32-1024 tokens")
    m=read_manifest(args.manifest.resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    from TTS_infer_pack.TTS import TTS,TTS_Config
    import soundfile as sf
    payload=torch.load(args.student,map_location="cpu",weights_only=False)
    teacher_cfg=torch.load(args.teacher,map_location="cpu",weights_only=False)["config"]
    from dataclasses import asdict
    if payload["format"]!="s1-next-v1" or payload["teacher_config"]!=teacher_cfg or payload["config"]!=asdict(NextConfig.from_teacher(teacher_cfg)):
        raise ValueError("Teacher/student configuration mismatch")
    student=S1Next(NextConfig(**payload["config"]))
    student.load_state_dict(payload["state_dict"],strict=True)
    del payload
    student=student.cuda().eval()
    rows=[]
    with tempfile.TemporaryDirectory(prefix="s1-next-waveform-") as tmp:
        cfg=TTS_Config({"custom":dict(TTS_Config.default_configs["v5dev"],
                       device="cuda:0",is_half=False,t2s_weights_path=str(args.teacher.resolve()),
                       vits_weights_path=str(args.s2.resolve()))})
        cfg.configs_path=str(Path(tmp)/"tts.yaml")
        engine=TTS(cfg)
        teacher=engine.t2s_model.model.eval()
        for ri,ref in enumerate(m["references"]):
            engine.set_ref_audio(ref["path"])
            for ti,target in enumerate(m["targets"]):
                seed=2027+ri*1009+ti
                for arm in ("teacher","student"):
                    row={"arm":arm,"reference_index":ri+1,"text_index":ti+1,"seed":seed}
                    try:
                        sr,wave,stats=synthesize(engine,student,teacher,ref,target,seed,args.max_tokens,arm)
                        name=f"{arm}-{ri+1:02}-{ti+1:02}.wav"
                        sf.write(args.output/name,wave,sr,subtype="PCM_16")
                        row.update(status="ok",file=name,**stats)
                    except Exception as exc:
                        row.update(status="failed",error_type=type(exc).__name__)
                    rows.append(row)
                    # Only aggregate identifiers in stdout; no audio or text.
                    print(json.dumps({k:v for k,v in row.items() if k!="file"},allow_nan=False),flush=True)
                    (args.output/"waveform-metrics.json").write_text(json.dumps(
                        {"rows":rows,"summary":summarize(rows)},allow_nan=False,indent=2))
    # Save only a private scoring request. Failed syntheses remain in denominator.
    successful=[r for r in rows if r["status"]=="ok"]
    scoring={"texts":m["targets"],"references":[r["path"] for r in m["references"]],
             "samples":[{"label":r["arm"],"path":str((args.output/r["file"]).resolve()),
                         "reference_index":r["reference_index"],"text_index":r["text_index"]}
                        for r in successful],"cpu_threads":2}
    (args.output/"scoring-input.json").write_text(json.dumps(scoring,ensure_ascii=False))
    if not successful:
        raise RuntimeError("No valid synthesis")

if __name__=="__main__":
    main()
