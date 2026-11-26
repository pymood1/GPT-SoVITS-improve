"""Round 5: prompt-aware distillation and independent bridge scale sweep."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path("/kaggle/tmp/GPT-SoVITS")
OUT=Path("/kaggle/working/s1-next-repair")
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
        _,rows=prepare_rows(split,count,folder/split,seen_audio,seen_text,folder/"hf-cache")
        (folder/(split+".json")).write_text(json.dumps([{"path":str(folder/split/"wav"/n),"text":t} for n,t,_ in rows],ensure_ascii=False))
    _,rows=prepare_rows("test",34,folder/"test",seen_audio,seen_text,folder/"hf-cache")
    # Exclude the first 29 previously evaluated utterances.
    fresh=rows[29:]
    manifest={"references":[{"path":str(folder/"test/wav"/n),"text":t} for n,t,_ in fresh[:2]],
              "targets":[t for _,t,_ in fresh[2:]]}
    path=folder/"manifest.json"
    path.write_text(json.dumps(manifest,ensure_ascii=False),encoding="utf-8")
    read_manifest(path)
    return path


def ensure_frontend_resources(env):
    """Preflight the exact missing NLTK resource before launching any GPU workers."""
    resource_dir=ROOT/"runtime/nltk_data"
    resource_dir.mkdir(parents=True,exist_ok=True)
    env["NLTK_DATA"]=os.pathsep.join(filter(None,(str(resource_dir),env.get("NLTK_DATA",""))))
    os.environ["NLTK_DATA"]=env["NLTK_DATA"]
    code="""
import os,nltk
from pathlib import Path
root=os.environ["NLTK_DATA"].split(os.pathsep)[0]
if root not in nltk.data.path:
    nltk.data.path.insert(0,root)
name="averaged_perceptron_tagger_eng"
try:
    nltk.data.find("taggers/"+name+"/")
except LookupError:
    if not nltk.download(name,download_dir=root,quiet=True,raise_on_error=True):
        raise RuntimeError("NLTK tagger download failed")
nltk.data.find("taggers/"+name+"/")
tags=nltk.pos_tag(["Korean","English","test"])
assert len(tags)==3
print("[preflight] English POS tagger verified",flush=True)
"""
    subprocess.run([sys.executable,"-u","-c",code],cwd=ROOT,env=env,check=True,timeout=180)


def error_signature(path):
    """Only exception class, stack locations and NLTK resource names, no raw corpus text."""
    import re
    if not path.is_file():
        return []
    raw=re.sub(r"\x1b\[[0-9;]*m","",path.read_text(errors="replace"))
    result=[]
    for part in raw.split("Traceback (most recent call last):")[1:]:
        frames=re.findall(r'File "([^"]+)", line (\d+), in ([^\n]+)',part)
        classes=re.findall(r"^([A-Za-z_][\w.]*Error):",part,re.M)
        resources=re.findall(r"Resource\s+([A-Za-z0-9_/-]+)\s+not found",part)
        result.append({"frames":[[Path(f).name,int(n),fn] for f,n,fn in frames[-8:]],
                       "exceptions":classes[:3],"nltk_resources":resources[:3]})
    return result[-3:]


def latest_progress(path):
    if not path.is_file():
        return None
    with path.open("rb") as f:
        f.seek(max(0,path.stat().st_size-16384))
        lines=f.read().decode(errors="replace").splitlines()
    for line in reversed(lines):
        try:
            row=json.loads(line)
            if not isinstance(row,dict):
                continue
            if row.get("event")=="pseudo_pair":
                return {k:row[k] for k in ("event","accepted","attempts")}
            if "epoch" in row and "validation" in row:
                return {"epoch":row["epoch"],"tokens":row.get("tokens",0)}
            if "arm" in row and "status" in row:
                return {k:row[k] for k in ("arm","reference_index","text_index","status")}
        except (ValueError,KeyError):
            continue
    return None


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
        report["stage"]="frontend_preflight"
        ensure_frontend_resources(env)
        report["nltk_pos_tagger"]="verified"
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
        report["stage"]="prompt_repair_and_bridge_sweep"
        common=[sys.executable,"-u",str(ROOT/"tools/evaluate_s1_next_waveforms.py"),
                "--teacher",str(teacher),"--s2",str(s2),"--manifest",str(manifest),"--max-tokens","512"]
        train=[sys.executable,"-u",str(ROOT/"tools/train_s1_next_prompt.py"),
               "--teacher",str(teacher),"--student",str(assets["student"]),"--s2",str(s2),
               "--train",str(manifest.parent/"train.json"),"--validation",str(manifest.parent/"validation.json"),
               "--output",str(OUT/"prompt"),"--train-pairs","96","--val-pairs","12","--epochs","12"]
        bridge=common+["--student",str(assets["student"]),"--aligned",str(assets["aligned"]),
                       "--global-mean",str(assets["global_mean"]),"--output",str(OUT/"gpu1"),
                       "--arms","teacher","aligned@0.05","aligned@0.15","aligned@0.35","global_mean@0.10"]
        workers=[]
        logs=[]
        try:
            for i,args in enumerate((train,bridge)):
                log=(OUT/f"worker{i}.log").open("w")
                logs.append(log)
                workers.append(subprocess.Popen(args,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(i)),stdout=log,stderr=subprocess.STDOUT))
            while any(p.poll() is None for p in workers):
                print("[repair]",json.dumps({"exit":[p.poll() for p in workers],
                    "prompt":latest_progress(OUT/"worker0.log"),"bridge":latest_progress(OUT/"worker1.log")}),flush=True)
                time.sleep(15)
        finally:
            for p in workers:
                if p.poll() is None:
                    p.terminate()
                    p.wait()
            for log in logs:
                log.close()
        report["worker_exit"]=[p.returncode for p in workers]
        report["diagnostics"]={str(p.relative_to(OUT)):error_signature(p) for p in (OUT/"worker0.log",OUT/"worker1.log",OUT/"gpu1/private-errors.log") if p.is_file()}
        metrics=OUT/"prompt/metrics.json"
        if metrics.is_file():
            report["prompt"]=json.loads(metrics.read_text())
        if workers[0].returncode==0:
            report["stage"]="repaired_student_waveforms"
            with (OUT/"student.log").open("w") as log:
                p=subprocess.run(common+["--student",str(OUT/"prompt/s1-next-best.pt"),
                         "--output",str(OUT/"gpu0"),"--arms","student"],cwd=ROOT,
                         env=dict(env,CUDA_VISIBLE_DEVICES="0"),stdout=log,stderr=subprocess.STDOUT,timeout=900)
            report["student_exit"]=p.returncode
            for file in (OUT/"student.log",OUT/"gpu0/private-errors.log"):
                if file.is_file():
                    report["diagnostics"][str(file.relative_to(OUT))]=error_signature(file)
        else:
            report["student_exit"]=-1
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
        report["expected_attempts"]=36
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
        report["status"]="completed" if all(p.returncode==0 for p in workers) and report["student_exit"]==0 and len(rows)==36 and all(v=="completed" for v in report["score_stages"].values()) else "partial"
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
