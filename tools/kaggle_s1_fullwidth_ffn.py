"""Round 12: full-width weight transfer, then matched waveform evaluation."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path("/kaggle/tmp/GPT-SoVITS")
OUT=Path("/kaggle/working/s1-fullwidth-ffn")
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


def previous_stack(root=Path("/kaggle/input")):
    kernel="gpt-sovits-s1-next-45812067222"
    matches=[p for p in root.rglob("s1-next-result.json") if kernel in p.parts]
    if len(matches)!=1:
        raise FileNotFoundError("Unique round 11 result required")
    report=json.loads(matches[0].read_text())
    if report["code_sha"]!="93659fe34513311597ec64081bb9d289bb3bd7b1" or report["status"]!="completed" or report["workers"]["ddp"]!=0 or report["training"]["accepted_layers"]!=[0,1,2,3,4,10,11] or report["training"]["selected"]["top1_agreement"]<.90:
        raise ValueError("Round 11 checkpoint provenance or preservation gate mismatch")
    path=matches[0].parent/"s1-joint-ffn/student/stack-best.pt"
    if not path.is_file() or path.stat().st_size<1000000:
        raise FileNotFoundError("Round 11 stack checkpoint missing")
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
                return {k:row[k] for k in ("event","split","accepted","attempts") if k in row}
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
        first=previous_stack()
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
        report["stage"]="fullwidth_ffn"
        teacher=ROOT/"GPT_SoVITS/pretrained_models/s1v3.ckpt"
        command=[sys.executable,"-u","-m","torch.distributed.run","--standalone","--nproc-per-node=2","--max-restarts=0",
                 str(ROOT/"tools/train_s1_fullwidth_ffn.py"),"--teacher",str(teacher),"--data",str(replay),"--first",str(first),
                 "--output",str(OUT/"student"),"--pairs","384","--epochs","3","--micro","4","--seconds","1200"]
        run_workers([command],[dict(env,CUDA_VISIBLE_DEVICES="0,1")],["ddp"],report)
        report["training"]=json.loads((OUT/"student/metrics.json").read_text())
        assert report["training"]["world_size"]==2
        candidate=OUT/"student/folded-stack.pt"
        assert candidate.is_file()
        report["stage"]="manifest"
        ensure_frontend_resources(env)
        from tools.kaggle_s1_next_convergence import prepare_manifest
        from tools.evaluate_s1_next_waveforms import summarize,read_manifest
        manifest,corpus=prepare_manifest(Path("/kaggle/tmp/s1-round12-corpus"))
        assert corpus["train"]==prior_report["corpus"]["train"]
        m=json.loads(manifest.read_text())
        m["targets"]=m["targets"][:2]
        manifest.write_text(json.dumps(m,ensure_ascii=False))
        read_manifest(manifest)
        report["corpus"]={**corpus,"test_targets":2}
        report["stage"]="waveform_eval"
        s2=ROOT/"GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"
        common=[sys.executable,"-u",str(ROOT/"tools/evaluate_s1_stack_waveforms.py"),
                "--teacher",str(teacher),"--s2",str(s2),"--student",str(candidate),
                "--prior-student",str(first),"--manifest",str(manifest),"--max-tokens","384"]
        envs=[dict(env,CUDA_VISIBLE_DEVICES=str(i)) for i in range(2)]
        cmds=[common+["--output",str(OUT/f"gpu{i}"),"--worker-index",str(i),"--worker-count","2"] for i in range(2)]
        run_workers(cmds,envs,["eval0","eval1"],report)
        rows,samples=[],[]
        request=None
        for i in range(2):
            folder=OUT/f"gpu{i}"
            rows.extend(json.loads((folder/"waveform-metrics.json").read_text())["rows"])
            req=json.loads((folder/"scoring-input.json").read_text())
            if request is None:
                request={k:v for k,v in req.items() if k!="samples"}
            samples.extend(req["samples"])
            report["diagnostics"][f"eval{i}_errors"]=error_signature(folder/"private-errors.log")
        report["waveforms"]=summarize(rows)
        report["expected_attempts"]=12
        report["actual_attempts"]=len(rows)
        report["errors"]=[{k:r[k] for k in ("arm","reference_index","text_index","error_type")} for r in rows if r["status"]=="failed"]
        if not samples:
            raise RuntimeError("No successful waveforms")
        report["stage"]="scoring"
        request["samples"]=samples
        source=OUT/"scoring-input.json"
        source.write_text(json.dumps(request,ensure_ascii=False))
        from tools.asr.fasterwhisper_asr import download_model
        download_model("large-v3")
        # Independent GPU scoring on two idle devices. Same backend within each comparison.
        # No CPU parity fallback; keep the session bounded if a scoring backend fails.
        stages=("asr","speaker")
        cmds=[["timeout","420",sys.executable,"-u",str(ROOT/"tools/score_s1_stack_waveforms.py"),
               str(source),stage,str(OUT/(stage+"-scores.json"))] for stage in stages]
        try:
            run_workers(cmds,envs,["score-asr","score-speaker"],report)
        except RuntimeError:
            pass
        report["score_stages"]={}
        for stage in stages:
            path=OUT/(stage+"-scores.json")
            if not path.is_file():
                report["score_stages"][stage]="failed"
                continue
            result=json.loads(path.read_text())
            assert len(result["scores"])==len(samples)
            report["score_stages"][stage]="completed"
            for arm,summary in report["waveforms"].items():
                vals=[v for r,v in zip(samples,result["scores"]) if r["label"]==arm]
                if not vals:
                    continue
                if stage=="asr":
                    errors=sum(v["char_errors"] for v in vals)
                    chars=sum(v["char_count"] for v in vals)
                    summary.update(cer=errors/max(1,chars),char_errors=errors,char_count=chars,asr_count=len(vals))
                else:
                    summary.update(speaker_cosine=sum(v["speaker_cosine"] for v in vals)/len(vals),speaker_count=len(vals))
            report.setdefault("score_backends",{})[stage]=result["backend"]
        report["status"]="completed" if len(rows)==12 and not report["errors"] and all(v=="completed" for v in report["score_stages"].values()) else "partial"
        report["stage"]="finished"
    except Exception as exc:
        report["error_type"]=type(exc).__name__
        import traceback
        (OUT/"private-error.log").write_text(traceback.format_exc(),encoding="utf-8")
        report.setdefault("diagnostics",{})["parent"]=error_signature(OUT/"private-error.log")
    finally:
        report["seconds"]=round(time.monotonic()-start,3)
        Path("/kaggle/working/s1-next-result.json").write_text(json.dumps(report,allow_nan=False,indent=2))
        print("Full-width FFN status",report["status"],report["stage"],flush=True)

if __name__=="__main__":
    main()
