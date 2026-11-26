"""Round 4: independent-reference waveform comparison, no retraining."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path("/kaggle/tmp/GPT-SoVITS")
OUT=Path("/kaggle/working/s1-next-waveforms")
SOURCE="pymood1/gpt-sovits-s1-next-45741706616"
SOURCE_SHA="bac60841fa7a560b8518bf2c4e18c654e70b36f8"


def prior_assets(kernel,root=Path("/kaggle/input")):
    if kernel!=SOURCE:
        raise ValueError("Unexpected prior run")
    slug=kernel.split("/")[-1]
    results=[p for p in root.rglob("s1-next-result.json") if slug in p.parts]
    if len(results)!=1:
        raise FileNotFoundError("Unique prior result required")
    report=json.loads(results[0].read_text())
    if report["code_sha"]!=SOURCE_SHA or report["status"]!="completed":
        raise ValueError("Prior run provenance mismatch")
    base=results[0].parent/"s1-next-pilot"
    assets={"student":base/"trajectory_kd/s1-next-best.pt",
            "aligned":base/"bridge/aligned-best.pt",
            "global_mean":base/"bridge/global_mean-best.pt"}
    for name,path in assets.items():
        if not path.is_file() or path.stat().st_size<10000:
            raise FileNotFoundError("Missing checkpoint "+name)
    return assets


def prepare_manifest(folder):
    from tools.prepare_s1_next_fleurs import prepare_rows
    from tools.evaluate_s1_next_waveforms import read_manifest
    folder.mkdir(parents=True,exist_ok=False)
    seen_audio,seen_text=set(),set()
    for split,count in (("train",384),("validation",32)):
        prepare_rows(split,count,folder/split,seen_audio,seen_text,folder/"hf-cache")
        shutil.rmtree(folder/split)
    _,rows=prepare_rows("test",29,folder/"test",seen_audio,seen_text,folder/"hf-cache")
    # Exclude all 24 previously evaluated utterances; no target waveform is passed to synthesis.
    fresh=rows[24:]
    manifest={"references":[{"path":str(folder/"test/wav"/n),"text":t} for n,t,_ in fresh[:2]],
              "targets":[t for _,t,_ in fresh[2:]]}
    path=folder/"manifest.json"
    path.write_text(json.dumps(manifest,ensure_ascii=False),encoding="utf-8")
    read_manifest(path)
    return path


def main():
    report={"code_sha":REQUEST["code_sha"],"status":"failed","stage":"mount"}
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    try:
        assets=prior_assets(REQUEST["source_kernel"])
        report["stage"]="clone"
        ROOT.mkdir(parents=True,exist_ok=False)
        subprocess.run(["git","init",str(ROOT)],check=True)
        subprocess.run(["git","-C",str(ROOT),"fetch","--depth","1","https://github.com/pymood1/GPT-SoVITS-improve.git",REQUEST["code_sha"]],check=True)
        subprocess.run(["git","-C",str(ROOT),"checkout","--detach","FETCH_HEAD"],check=True)
        if subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()!=REQUEST["code_sha"]:
            raise ValueError("Pinned code mismatch")
        report["stage"]="runtime_setup"
        env=dict(os.environ,HF_HOME="/kaggle/tmp/huggingface",TMPDIR="/kaggle/tmp",PYTHONUNBUFFERED="1",
                 PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/"GPT_SoVITS")]))
        setup=ROOT/"tools/colab_fast_setup.py"
        original=setup.read_bytes()
        try:
            setup.write_text(original.decode().replace("/content/","/kaggle/tmp/"),encoding="utf-8")
            subprocess.run([sys.executable,"-u",str(setup)],cwd=ROOT,env=env,check=True)
        finally:
            setup.write_bytes(original)
        shutil.rmtree("/kaggle/tmp/gpt_sovits_downloads",ignore_errors=True)
        gpus=subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True).splitlines()
        if len(gpus)!=2 or any("T4" not in g for g in gpus):
            raise RuntimeError("Expected two T4 GPUs")
        report["gpu_count"]=len(gpus)
        sys.path[:0]=[str(ROOT),str(ROOT/"GPT_SoVITS")]
        os.chdir(ROOT)
        report["stage"]="manifest"
        manifest=prepare_manifest(Path("/kaggle/tmp/s1-next-waveform-corpus"))
        teacher=ROOT/"GPT_SoVITS/pretrained_models/s1v3.ckpt"
        s2=ROOT/"GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"
        report["stage"]="waveforms"
        workers=[]
        logs=[]
        try:
            for i in range(2):
                log=(OUT/f"gpu{i}.log").open("w")
                logs.append(log)
                args=[sys.executable,"-u",str(ROOT/"tools/evaluate_s1_next_waveforms.py"),
                      "--teacher",str(teacher),"--student",str(assets["student"]),"--s2",str(s2),
                      "--aligned",str(assets["aligned"]),"--global-mean",str(assets["global_mean"]),
                      "--manifest",str(manifest),"--output",str(OUT/f"gpu{i}"),
                      "--worker-index",str(i),"--worker-count","2","--max-tokens","512"]
                workers.append(subprocess.Popen(args,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(i)),stdout=log,stderr=subprocess.STDOUT))
            while any(p.poll() is None for p in workers):
                print("[waveforms] worker states",[p.poll() for p in workers],flush=True)
                time.sleep(15)
        finally:
            for p in workers:
                if p.poll() is None:
                    p.terminate()
                    p.wait()
            for log in logs:
                log.close()
        report["worker_exit"]=[p.returncode for p in workers]
        from tools.evaluate_s1_next_waveforms import summarize
        rows=[]
        samples=[]
        request=None
        for i in range(2):
            folder=OUT/f"gpu{i}"
            path=folder/"waveform-metrics.json"
            if path.is_file():
                rows.extend(json.loads(path.read_text())["rows"])
            path=folder/"scoring-input.json"
            if path.is_file():
                req=json.loads(path.read_text())
                if request is None:
                    request={k:v for k,v in req.items() if k!="samples"}
                samples.extend(req["samples"])
        report["waveforms"]=summarize(rows)
        report["expected_attempts"]=24
        report["actual_attempts"]=len(rows)
        report["errors"]=[{"arm":r["arm"],"reference_index":r["reference_index"],"text_index":r["text_index"],"error_type":r["error_type"]} for r in rows if r["status"]=="failed"]
        if not samples or request is None:
            raise RuntimeError("No successful waveforms")
        report["stage"]="scoring"
        request["samples"]=samples
        (OUT/"scoring-input.json").write_text(json.dumps(request,ensure_ascii=False),encoding="utf-8")
        # Reuse the validated project scoring stage including GPU/CPU parity fallback.
        from tools.asr.fasterwhisper_asr import download_model
        download_model("large-v3")
        from tools.kaggle_quality_listening import run_score_stage
        report["score_stages"]={}
        for stage in ("asr","speaker"):
            result=run_score_stage(OUT,stage,request)
            report["score_stages"][stage]="completed" if result else "failed"
            if result:
                for label,summary in report["waveforms"].items():
                    scores=[s for r,s in zip(samples,result["scores"]) if r["label"]==label]
                    if not scores:
                        continue
                    if stage=="asr":
                        errors=sum(s["char_errors"] for s in scores)
                        chars=sum(s["char_count"] for s in scores)
                        summary.update(cer=errors/max(1,chars),char_errors=errors,char_count=chars,asr_count=len(scores))
                    else:
                        summary.update(speaker_cosine=sum(s["speaker_cosine"] for s in scores)/len(scores),speaker_count=len(scores))
        report["status"]="completed" if all(p.returncode==0 for p in workers) and len(rows)==24 and all(v=="completed" for v in report["score_stages"].values()) else "partial"
        report["stage"]="finished"
    except Exception as exc:
        report["error_type"]=type(exc).__name__
        import traceback
        (OUT/"private-error.log").write_text(traceback.format_exc(),encoding="utf-8")
    finally:
        report["seconds"]=round(time.monotonic()-start,3)
        Path("/kaggle/working/s1-next-result.json").write_text(json.dumps(report,allow_nan=False,indent=2),encoding="utf-8")
        print("Waveform status",report["status"],report["stage"],flush=True)

if __name__=="__main__":
    main()
