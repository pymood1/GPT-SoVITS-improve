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
from AR.models.s1_next import NextConfig,S1Next,SemanticResidualBridge
from module.s1_next_bridge import FrozenV5Bridge
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


def features(engine,text,language="all_ko"):
    if text[-1] not in ".!?。！？":
        text+="."
    phones,bert,_=engine.text_preprocessor.segment_and_extract_feature_for_text(
        text,language,engine.configs.version)
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
def synthesize(engine,student,teacher,ref,target,seed,cap,arm,bridges=None,cache=None,prior_student=None):
    import numpy as np
    px,pb=features(engine,ref["text"])
    tx,tb=features(engine,target)
    x=torch.cat((px,tx),dim=1)
    b=torch.cat((pb,tb),dim=2)
    engine.prompt_cache["phones"]=px[0].tolist()
    prompt=engine.prompt_cache["prompt_semantic"].reshape(1,-1).long()
    started=time.monotonic()
    teacher_arm=arm not in ("student","prior_student")
    key=(seed,"teacher" if teacher_arm else arm)
    if cache is not None and key in cache:
        ids,hidden,generate_seconds=cache[key]
    else:
        if teacher_arm:
            ids=teacher_generate(teacher,x,b,prompt,cap=cap,top_k=15,penalty=1.35,seed=seed)
            hidden=None
        else:
            g=torch.Generator(device=x.device).manual_seed(seed)
            active_student=prior_student if arm=="prior_student" else student
            ids,hidden=active_student.generate(x,b,prompt,max_new=cap,min_new=10,top_k=15,
                                       repetition_penalty=1.35,generator=g)
        generate_seconds=time.monotonic()-started
        if cache is not None:
            cache[key]=(ids,hidden,generate_seconds)
    stats=token_stats(ids,cap)
    stats["s1_seconds"]=generate_seconds
    if ids.numel()==0 or int(ids.max())>=student.cfg.codes:
        raise ValueError("Empty or invalid generated semantic tokens")
    # Equal acoustic sampling seed, identical frozen S2 and vocoder settings.
    torch.manual_seed(seed+100000)
    if arm in ("teacher","student","prior_student"):
        wave=engine.using_vocoder_synthesis(ids[:,None,:],tx,sample_steps=32,cfg_rate=1.30)
    else:
        from TTS_infer_pack.TTS import norm_spec,denorm_spec,mel_fn_v4,resample
        from module.models_v5 import synthesize_v5_mel
        model,global_mean=bridges[arm.split("@")[0]]
        mode=arm.split("@")[0]
        scale=float(arm.split("@")[1]) if "@" in arm else 1.
        raw=engine.prompt_cache["refer_spec"][0]
        refer=(raw[0] if isinstance(raw,tuple) else raw).float()
        if mode=="aligned":
            y=torch.cat((prompt,ids),dim=1)
            xy,mask,_=teacher.make_input_data(x,torch.tensor([x.size(1)],device=x.device),
                                            y,torch.tensor([y.size(1)],device=x.device),b)
            h,_=teacher.h((xy,None),mask=mask)
            hidden=h[:,-ids.size(1):]
        elif mode=="global_mean":
            hidden=global_mean.to(x.device).expand(1,ids.size(1),-1)
        else:
            raise ValueError("Unknown arm")
        fea_ref,ge=engine.vits_model.decode_encp(prompt[:,None,:],px,refer)
        original,_=engine.vits_model.decode_encp(ids[:,None,:],tx,refer,ge,1.)
        hard,_=model.features(ids[:,None,:],hidden,tx,refer,enabled=False)
        torch.testing.assert_close(hard,original,rtol=1e-5,atol=1e-5)
        fea,_=model.features(ids[:,None,:],hidden,tx,refer,residual_scale=scale)
        ref_audio=engine.prompt_cache["raw_audio"].to(x.device).float()
        if ref_audio.shape[0]==2:
            ref_audio=ref_audio.mean(0,keepdim=True)
        sr0=engine.prompt_cache["raw_sr"]
        if sr0!=32000:
            ref_audio=resample(ref_audio,sr0,32000,x.device)
        mel=norm_spec(mel_fn_v4(ref_audio))
        torch.manual_seed(seed+100000)
        mel=synthesize_v5_mel(engine.vits_model,fea_ref,fea,mel,32,1.30)
        wave=engine.vocoder(denorm_spec(mel))[0,0]
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
    for arm in sorted({r["arm"] for r in rows}):
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
    p.add_argument("--prior-student",type=Path)
    p.add_argument("--s2",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-tokens",type=int,default=512)
    p.add_argument("--aligned",type=Path)
    p.add_argument("--global-mean",type=Path)
    p.add_argument("--arms",nargs="+")
    p.add_argument("--worker-index",type=int,default=0)
    p.add_argument("--worker-count",type=int,default=1)
    args=p.parse_args()
    if not 0<=args.worker_index<args.worker_count:
        p.error("Invalid worker index")
    if bool(args.aligned)!=bool(args.global_mean):
        p.error("Provide both bridge checkpoints or neither")
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
    prior_student=None
    if args.prior_student:
        previous=torch.load(args.prior_student,map_location="cpu",weights_only=False)
        if previous["format"]!="s1-next-v1" or previous["teacher_config"]!=teacher_cfg or previous["config"]!=asdict(student.cfg):
            raise ValueError("Prior student checkpoint mismatch")
        prior_student=S1Next(student.cfg)
        prior_student.load_state_dict(previous["state_dict"],strict=True)
        del previous
        prior_student=prior_student.cuda().eval()
    rows=[]
    with tempfile.TemporaryDirectory(prefix="s1-next-waveform-") as tmp:
        cfg=TTS_Config({"custom":dict(TTS_Config.default_configs["v5dev"],
                       device="cuda:0",is_half=False,t2s_weights_path=str(args.teacher.resolve()),
                       vits_weights_path=str(args.s2.resolve()))})
        cfg.configs_path=str(Path(tmp)/"tts.yaml")
        engine=TTS(cfg)
        teacher=engine.t2s_model.model.eval()
        bridges={}
        if args.aligned:
            for mode,path in (("aligned",args.aligned),("global_mean",args.global_mean)):
                saved=torch.load(path,map_location="cpu",weights_only=True)
                if saved["condition"]!=mode or saved["initialization"]!="zero_projection" or saved["source"]!="original-s1-teacher":
                    raise ValueError("Bridge checkpoint contract mismatch")
                adapter=SemanticResidualBridge(student.cfg.dim,768,128,initialization="zero_projection")
                adapter.load_state_dict(saved["adapter"],strict=True)
                bridges[mode]=(FrozenV5Bridge(engine.vits_model,adapter.cuda()).eval(),saved["global_mean"])
        arms=args.arms or ["teacher","student",*(["prior_student"] if prior_student else []),*bridges]
        if len(set(arms))!=len(arms) or any(a not in ("teacher","student") and not (a=="prior_student" and prior_student is not None) and a.split("@")[0] not in bridges for a in arms):
            raise ValueError("Invalid arm selection")
        for ri,ref in enumerate(m["references"]):
            if ri%args.worker_count!=args.worker_index:
                continue
            engine.set_ref_audio(ref["path"])
            cache={}
            for ti,target in enumerate(m["targets"]):
                seed=2027+ri*1009+ti
                for arm in arms:
                    row={"arm":arm,"reference_index":ri+1,"text_index":ti+1,"seed":seed}
                    try:
                        sr,wave,stats=synthesize(engine,student,teacher,ref,target,seed,args.max_tokens,arm,bridges,cache,prior_student)
                        name=f"{arm.replace(chr(64),chr(45))}-{ri+1:02}-{ti+1:02}.wav"
                        sf.write(args.output/name,wave,sr,subtype="PCM_16")
                        row.update(status="ok",file=name,**stats)
                    except Exception as exc:
                        import traceback
                        with (args.output/"private-errors.log").open("a",encoding="utf-8") as log:
                            log.write(f"{arm}/{ri+1}/{ti+1}:\n"+traceback.format_exc()+"\n")
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
