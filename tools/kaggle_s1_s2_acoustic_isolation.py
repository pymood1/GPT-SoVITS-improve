"""Frozen acoustic isolation launcher. Reuse exact listening references; no training."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path("/kaggle/tmp/GPT-SoVITS")
OUT=Path("/kaggle/working/acoustic-workers")
SOURCE="pymood1/gpt-sovits-s1-next-45760100606"
SOURCE_SHA="7955f9e5279a96177132a182cc65e3ddf6a9e288"


def prior_student(kernel,root=Path("/kaggle/input")):
    if kernel!=SOURCE:
        raise ValueError("Unexpected source kernel")
    results=[p for p in root.rglob("s1-next-result.json") if kernel.split("/")[-1] in p.parts]
    if len(results)!=1:
        raise FileNotFoundError("Unique prior result required")
    report=json.loads(results[0].read_text())
    if report["code_sha"]!=SOURCE_SHA or report["status"]!="completed" or report["workers"]["ddp"]!=0:
        raise ValueError("Prior run provenance mismatch")
    base=results[0].parent/"s1-next-scale"
    checkpoint=base/"student/s1-next-best.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size<1000000:
        raise FileNotFoundError("Prior student checkpoint missing")
    for rank in range(2):
        for name in ("train.pt","validation.pt"):
            if not (base/f"pseudo{rank}"/name).is_file():
                raise FileNotFoundError("Missing teacher replay cache")
    return checkpoint,base,report


def previous_fullwidth(root=Path("/kaggle/input")):
    kernel="gpt-sovits-s1-next-45815400064"
    matches=[p for p in root.rglob("s1-next-result.json") if kernel in p.parts]
    if len(matches)!=1:
        raise FileNotFoundError("Unique round 12 result required")
    report=json.loads(matches[0].read_text())
    if report["code_sha"]!="0e4eb677fd3bf76c419a7e84656c5e7b25781d18" or report["status"] not in ("partial","completed") or not report["training"]["export_gate_met"] or report["actual_attempts"]!=12 or report["errors"]:
        raise ValueError("Round 12 provenance or quality gate mismatch")
    base=matches[0].parent/"s1-fullwidth-ffn"
    stack=base/"student/folded-stack.pt"
    if not stack.is_file():
        raise FileNotFoundError("Missing exported round 12 stack")
    return base,stack,report


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
                return {k:row[k] for k in ("event","split","accepted","attempts") if k in row}
            if row.get("event")=="asr_progress":
                return {k:row[k] for k in ("event","backend","completed","total")}
            if row.get("event")=="cache_trial":
                return {k:row[k] for k in ("event","index","tokens","models")}
            if row.get("event") in ("layer_epoch","layer_result"):
                return {k:row[k] for k in ("event","stage","scale","group","epoch","tokens","accepted","accepted_count","validation","gate_streak") if k in row}
            if row.get("event")=="train_step":
                return {k:row[k] for k in ("event","epoch","step","tokens")}
            if "epoch" in row and "validation" in row:
                return {"epoch":row["epoch"],"tokens":row.get("tokens",0),"validation":row["validation"],"lr":row.get("lr"),"gate_streak":row.get("gate_streak",0)}
            if "arm" in row and "status" in row:
                return {k:row[k] for k in ("arm","reference_index","text_index","status")}
        except (ValueError,KeyError):
            continue
    return None



def run_workers(commands,envs,labels,report):
    workers=[]
    logs=[]
    try:
        for command,env,label in zip(commands,envs,labels):
            log=(OUT/(label+".log")).open("w")
            logs.append(log)
            workers.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in workers):
            progress={}
            for label,p in zip(labels,workers):
                progress[label]={"exit":p.poll(),"latest":latest_progress(OUT/(label+".log"))}
            print("[scale]",json.dumps(progress),flush=True)
            time.sleep(15)
    finally:
        for p in workers:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        for log in logs:
            log.close()
    exits=[p.returncode for p in workers]
    report.setdefault("workers",{}).update(dict(zip(labels,exits)))
    report.setdefault("diagnostics",{}).update({label:error_signature(OUT/(label+".log")) for label in labels})
    if any(code!=0 for code in exits):
        raise RuntimeError("Worker failed")


def previous_rms(root=Path("/kaggle/input")):
    kernel="gpt-sovits-s1-next-45828829274"
    matches=[p for p in root.rglob("s1-next-result.json") if kernel in p.parts]
    assert len(matches)==1
    report=json.loads(matches[0].read_text())
    assert report["code_sha"]=="d209ad8ef683801bf7c1bd0758bd7515537a2c18"
    assert report["status"]=="completed" and report["training"]["full_rms"]
    assert report["training"]["selected"]["top1_agreement"]>=.95
    path=matches[0].parent/"s1-affine-rms/student/norm-best.pt"
    assert path.is_file() and path.stat().st_size>1000000
    return path


def fresh_manifest(folder):
    from tools.prepare_s1_next_fleurs import prepare_rows
    from tools.evaluate_s1_next_waveforms import read_manifest
    folder.mkdir(parents=True,exist_ok=False)
    seen_audio,seen_text=set(),set()
    cache=folder/"hf-cache"
    def select(split,count,name,minimum=None):
        _,rows=prepare_rows(split,count,folder/name,seen_audio,seen_text,cache,min_count=minimum)
        return [{"path":str(folder/name/"wav"/n),"text":t} for n,t,_ in rows]
    # Reconstruct exact earlier selection order including round 6 test reserve.
    train=select("train",384,"train")
    validation=select("validation",32,"validation")
    test=select("test",40,"test-reserved")
    train+=select("train",256,"train-extra",0)
    old=select("test",6,"previous-test-fresh")
    assert len(train)==442 and len(validation)==32 and len(test)==40 and len(old)==6
    new=select("test",6,"round15-test")
    path=folder/"manifest.json"
    path.write_text(json.dumps({"references":new[:2],"targets":[r["text"] for r in new[2:]]},ensure_ascii=False))
    read_manifest(path)
    return path,{"train":len(train),"validation":len(validation),"reserved_test":len(test)+len(old),
                 "test_references":2,"test_targets":4,"dedup":"NFKC text and decoded audio SHA256"}


def main():
    report={"code_sha":REQUEST["code_sha"],"status":"failed","stage":"mount","training_performed":False,"source_run":45840975381}
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    listen=Path("/kaggle/working/acoustic")
    listen.mkdir()
    try:
        kernel="gpt-sovits-s1-next-45840975381"
        assert REQUEST["source_kernel"]=="pymood1/"+kernel
        paths=[p for p in Path("/kaggle/input").rglob("s1-next-result.json") if kernel in p.parts]
        assert len(paths)==1
        source=json.loads(paths[0].read_text())
        assert source["code_sha"]=="44b7d4787507da12b710c5d088d0299c2487ce32" and source["status"]=="completed" and source["audio_count"]==32
        previous=paths[0].parent/"listen"
        report["stage"]="clone"
        ROOT.mkdir(parents=True,exist_ok=False)
        subprocess.run(["git","init",str(ROOT)],check=True)
        subprocess.run(["git","-C",str(ROOT),"fetch","--depth","1","https://github.com/pymood1/GPT-SoVITS-improve.git",REQUEST["code_sha"]],check=True)
        subprocess.run(["git","-C",str(ROOT),"checkout","--detach","FETCH_HEAD"],check=True)
        assert subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()==REQUEST["code_sha"]
        env=dict(os.environ,HF_HOME="/kaggle/tmp/huggingface",TMPDIR="/kaggle/tmp",PYTHONUNBUFFERED="1",
                 PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/"GPT_SoVITS")]),
                 OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",NCCL_IB_DISABLE="1")
        report["stage"]="runtime_setup"
        setup=ROOT/"tools/colab_fast_setup.py"
        original=setup.read_bytes()
        try:
            setup.write_text(original.decode().replace("/content/","/kaggle/tmp/"),encoding="utf-8")
            subprocess.run([sys.executable,"-u",str(setup)],cwd=ROOT,env=env,check=True)
        finally:
            setup.write_bytes(original)
        shutil.rmtree("/kaggle/tmp/gpt_sovits_downloads",ignore_errors=True)
        gpus=subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True).splitlines()
        assert len(gpus)==2 and all("T4" in x for x in gpus)
        report["gpu_count"]=2
        sys.path[:0]=[str(ROOT),str(ROOT/"GPT_SoVITS")]
        os.chdir(ROOT)
        report["stage"]="manifest"
        ensure_frontend_resources(env)
        manifest,report["corpus"]=fresh_manifest(Path("/kaggle/tmp/s1-round15-corpus"))
        m=json.loads(manifest.read_text())
        old=json.loads((previous/"texts.json").read_text())
        assert old["references"]==[r["text"] for r in m["references"]] and old["targets"]==m["targets"]
        import hashlib
        for i,r in enumerate(m["references"],1):
            original=previous/f"reference-{i:02}.wav"
            assert hashlib.sha256(original.read_bytes()).digest()==hashlib.sha256(Path(r["path"]).read_bytes()).digest()
        lst=manifest.parent/"round15-test/input.list"
        entries=[line.split("|",3) for line in lst.read_text().splitlines() if line]
        assert len(entries)==6 and [row[3] for row in entries[2:]]==m["targets"]
        m["targets"]=[{"path":row[0],"text":row[3]} for row in entries[2:]]
        manifest.write_text(json.dumps(m,ensure_ascii=False))
        (listen/"README.txt").write_text("Frozen S1/S2 acoustic isolation. Both references use text01. Raw WAVs plus level-matched copies (bounded gain, no denoising). Compare target-real vs target-true-mel-vocoder, oracle-s2 vs teacher-baseline, teacher-baseline vs teacher-cfg0/cache-off/ref-gain. The oracle has real target semantic codes, not a normal TTS output. Quiet percentile is not a noise classifier. No training. Mel NPZ files use pre-vocoder log-mel.",encoding="utf-8")
        report["stage"]="acoustic_isolation"
        common=[sys.executable,"-u",str(ROOT/"tools/s1_s2_acoustic_isolation.py"),
            "--teacher",str(ROOT/"GPT_SoVITS/pretrained_models/s1v3.ckpt"),
            "--s2",str(ROOT/"GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"),"--manifest",str(manifest)]
        cmds=[common+["--output",str(listen/f"ref{i+1:02}"),"--worker-index",str(i)] for i in range(2)]
        try:
            run_workers(cmds,[dict(env,CUDA_VISIBLE_DEVICES=str(i)) for i in range(2)],["eval0","eval1"],report)
        except RuntimeError:
            pass
        report["rows"]=[]
        for i in range(2):
            folder=listen/f"ref{i+1:02}"
            if (folder/"metrics.json").is_file():
                report["rows"].extend(json.loads((folder/"metrics.json").read_text()))
            oldwav=previous/f"ref{i+1:02}-text01-teacher.wav"
            if oldwav.is_file() and folder.is_dir():
                shutil.copyfile(oldwav,folder/"previous-teacher.wav")
        report["expected_cases"]=20
        report["actual_cases"]=len(report["rows"])
        report["status"]="completed" if len(report["rows"])==20 and all(v==0 for v in report["workers"].values()) else "partial" if report["rows"] else "failed"
        report["stage"]="finished"
    except Exception as exc:
        report["error_type"]=type(exc).__name__
        import traceback
        (OUT/"private-error.log").write_text(traceback.format_exc(),encoding="utf-8")
        report.setdefault("diagnostics",{})["parent"]=error_signature(OUT/"private-error.log")
    finally:
        report["seconds"]=round(time.monotonic()-start,3)
        Path("/kaggle/working/s1-next-result.json").write_text(json.dumps(report,allow_nan=False,indent=2))
        print("Acoustic isolation status",report["status"],report.get("actual_cases",0),flush=True)


if __name__=="__main__":
    main()
