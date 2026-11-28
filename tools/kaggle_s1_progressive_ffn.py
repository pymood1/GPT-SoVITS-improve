"""Round 9: first-layer-only FFN replacement using round 6 cache."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path("/kaggle/tmp/GPT-SoVITS")
OUT=Path("/kaggle/working/s1-progressive-ffn")
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
            if any(p.poll() not in (0,None) for p in workers):
                for p in workers:
                    if p.poll() is None:
                        p.terminate()
                break
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


def main():
    report={"code_sha":REQUEST["code_sha"],"status":"failed","stage":"mount"}
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    try:
        previous,replay,prior_report=prior_student(REQUEST["source_kernel"])
        report["stage"]="clone"
        ROOT.mkdir(parents=True,exist_ok=False)
        subprocess.run(["git","init",str(ROOT)],check=True)
        subprocess.run(["git","-C",str(ROOT),"fetch","--depth","1","https://github.com/pymood1/GPT-SoVITS-improve.git",REQUEST["code_sha"]],check=True)
        subprocess.run(["git","-C",str(ROOT),"checkout","--detach","FETCH_HEAD"],check=True)
        if subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()!=REQUEST["code_sha"]:
            raise ValueError("Pinned code mismatch")
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
        if len(gpus)!=2 or any("T4" not in x for x in gpus):
            raise RuntimeError("Expected T4 x2")
        report["gpu_count"]=2
        sys.path[:0]=[str(ROOT),str(ROOT/"GPT_SoVITS")]
        os.chdir(ROOT)
        report["stage"]="progressive_ffn"
        teacher=ROOT/"GPT_SoVITS/pretrained_models/s1v3.ckpt"
        command=[sys.executable,"-u","-m","torch.distributed.run","--standalone","--nproc-per-node=2","--max-restarts=0",
                 str(ROOT/"tools/train_s1_progressive_ffn.py"),"--teacher",str(teacher),"--data",str(replay),
                 "--output",str(OUT/"student"),"--pairs","512","--epochs","8","--batch","1024","--seconds","600"]
        run_workers([command],[dict(env,CUDA_VISIBLE_DEVICES="0,1")],["ddp"],report)
        report["training"]=json.loads((OUT/"student/metrics.json").read_text())
        assert report["training"]["world_size"]==2
        assert (OUT/"student/ffn-best.pt").is_file()
        report["status"]="completed"
        report["stage"]="finished"
    except Exception as exc:
        report["error_type"]=type(exc).__name__
        import traceback
        (OUT/"private-error.log").write_text(traceback.format_exc(),encoding="utf-8")
        report.setdefault("diagnostics",{})["parent"]=error_signature(OUT/"private-error.log")
    finally:
        report["seconds"]=round(time.monotonic()-start,3)
        Path("/kaggle/working/s1-next-result.json").write_text(json.dumps(report,allow_nan=False,indent=2))
        print("Progressive FFN status",report["status"],report["stage"],flush=True)

if __name__=="__main__":
    main()
