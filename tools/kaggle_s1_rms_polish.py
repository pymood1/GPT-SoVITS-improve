"""Round 15: refine accepted RMS model, compare cached speed and fresh waveforms."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path("/kaggle/tmp/GPT-SoVITS")
OUT=Path("/kaggle/working/s1-rms-polish")
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
    report={"code_sha":REQUEST["code_sha"],"status":"failed","stage":"mount"}
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    try:
        _,replay,_=prior_student(REQUEST["source_kernel"])
        source,stack,prior=previous_fullwidth()
        resume=previous_rms()
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
        # Add installed CUDA wheel library paths before importing CTranslate2 in subprocess.
        import site
        libs=[]
        for folder in site.getsitepackages():
            libs.extend(str(p) for p in sorted(Path(folder).glob("nvidia/*/lib")) if p.is_dir())
        libs.extend(str(p) for p in (Path("/usr/local/cuda/lib64"),Path("/usr/lib/x86_64-linux-gnu")) if p.is_dir())
        env["LD_LIBRARY_PATH"]=os.pathsep.join(dict.fromkeys(libs+[p for p in env.get("LD_LIBRARY_PATH","").split(os.pathsep) if p]))
        report["stage"]="polish"
        teacher=ROOT/"GPT_SoVITS/pretrained_models/s1v3.ckpt"
        tool=str(ROOT/"tools/train_s1_rms_polish.py")
        common=["--teacher",str(teacher),"--data",str(replay),"--stack",str(stack),"--resume",str(resume)]
        cmd=[sys.executable,"-u","-m","torch.distributed.run","--standalone","--nproc-per-node=2","--max-restarts=0",
             tool,"--mode","train",*common,"--output",str(OUT/"student"),"--pairs","512","--epochs","8","--seconds","720"]
        run_workers([cmd],[dict(env,CUDA_VISIBLE_DEVICES="0,1")],["ddp"],report)
        report["training"]=json.loads((OUT/"student/metrics.json").read_text())
        refined=OUT/"student/norm-best.pt"
        assert refined.is_file() and report["training"]["world_size"]==2
        report["stage"]="cache_audit"
        envs=[dict(env,CUDA_VISIBLE_DEVICES=str(i)) for i in range(2)]
        cmds=[[sys.executable,"-u",tool,"--mode","benchmark",*common,"--refined",str(refined),
               "--output",str(OUT/f"cache{i}"),"--rank",str(i)] for i in range(2)]
        run_workers(cmds,envs,["cache0","cache1"],report)
        rows=[]
        for i in range(2):
            rows.extend(json.loads((OUT/f"cache{i}/metrics.json").read_text()))
        assert len(rows)==24 and len({r["index"] for r in rows})==24
        total=sum(r["tokens"] for r in rows)
        report["cache"]={}
        for label in ("teacher","round12","round14","student"):
            metrics=[r["models"][label] for r in rows]
            full=sum(r["full_seconds"] for r in metrics)
            cached=sum(r["cached_seconds"] for r in metrics)
            sampled=[r for r in metrics if "sampled_equal" in r]
            report["cache"][label]={"pairs":len(rows),"tokens":total,"full_seconds":full,"cached_seconds":cached,
                "speedup":full/cached,"full_tokens_per_second":total/full,"cached_tokens_per_second":total/cached,
                "max_abs":max(r["max_abs"] for r in metrics),
                "mean_abs":sum(r["mean_abs"]*row["tokens"] for r,row in zip(metrics,rows))/total,
                "parity_pairs":sum(r["parity"] for r in metrics),
                "top1_agreement":sum(r["top1_matches"] for r in metrics)/total,
                "max_cache_bytes":max(r["cache_bytes"] for r in metrics),
                "sampled_trials":len(sampled),"sampled_exact":sum(r["sampled_equal"] for r in sampled)}
        report["cache_gate_met"]=all(v["parity_pairs"]==24 and v["top1_agreement"]>=.999 and v["sampled_exact"]==v["sampled_trials"]==4 for v in report["cache"].values())
        (OUT/"cache-metrics.json").write_text(json.dumps(report["cache"],indent=2))
        report["stage"]="manifest"
        ensure_frontend_resources(env)
        manifest,report["corpus"]=fresh_manifest(Path("/kaggle/tmp/s1-round15-corpus"))
        report["stage"]="waveform_eval"
        from tools.evaluate_s1_next_waveforms import summarize
        s2=ROOT/"GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"
        common=[sys.executable,"-u",str(ROOT/"tools/evaluate_s1_rms_waveforms.py"),
                "--teacher",str(teacher),"--stack",str(stack),"--resume",str(resume),"--refined",str(refined),
                "--s2",str(s2),"--manifest",str(manifest),"--max-tokens","384"]
        cmds=[common+["--output",str(OUT/f"gpu{i}"),"--worker-index",str(i),"--worker-count","2"] for i in range(2)]
        run_workers(cmds,envs,["eval0","eval1"],report)
        rows=[];samples=[];request=None
        for i in range(2):
            folder=OUT/f"gpu{i}"
            rows.extend(json.loads((folder/"waveform-metrics.json").read_text())["rows"])
            req=json.loads((folder/"scoring-input.json").read_text())
            if request is None:
                request={k:v for k,v in req.items() if k!="samples"}
            else:
                assert request["texts"]==req["texts"] and request["references"]==req["references"]
            samples.extend(req["samples"])
            report["diagnostics"][f"eval{i}_errors"]=error_signature(folder/"private-errors.log")
        report["waveforms"]=summarize(rows)
        report["expected_attempts"]=32;report["actual_attempts"]=len(rows)
        report["errors"]=[{k:r[k] for k in ("arm","reference_index","text_index","error_type")} for r in rows if r["status"]=="failed"]
        assert len(rows)==32 and len(samples)==32 and not report["errors"]
        assert len({(r["label"],r["reference_index"],r["text_index"]) for r in samples})==32
        report["stage"]="scoring"
        request["samples"]=samples
        req=OUT/"scoring-input.json";req.write_text(json.dumps(request,ensure_ascii=False))
        from tools.asr.fasterwhisper_asr import download_model
        download_model("large-v3")
        asr=OUT/"asr-fp32.json";speaker=OUT/"speaker-scores.json"
        cmds=[["timeout","600",sys.executable,"-u",str(ROOT/"tools/score_s1_fp32_asr.py"),
               "--source",str(req),"--output",str(asr),"--backend","ct2","--expected-count","32"],
              ["timeout","420",sys.executable,"-u",str(ROOT/"tools/score_s1_stack_waveforms.py"),str(req),"speaker",str(speaker)]]
        try:
            run_workers(cmds,envs,["asr-fp32","speaker"],report)
        except RuntimeError:
            pass
        report["score_stages"]={}
        for stage,target in (("asr",asr),("speaker",speaker)):
            if not target.is_file():
                report["score_stages"][stage]="failed";continue
            result=json.loads(target.read_text())
            assert len(result["scores"])==32
            if stage=="asr":
                assert result["backend"]=="ct2_cuda_float32"
            report.setdefault("score_backends",{})[stage]=result["backend"]
            report["score_stages"][stage]="completed"
            for label,summary in report["waveforms"].items():
                vals=[v for r,v in zip(samples,result["scores"]) if r["label"]==label]
                assert len(vals)==8
                if stage=="asr":
                    errors=sum(v["char_errors"] for v in vals);chars=sum(v["char_count"] for v in vals)
                    summary.update(cer=errors/max(1,chars),char_errors=errors,char_count=chars,asr_count=len(vals))
                else:
                    summary.update(speaker_cosine=sum(v["speaker_cosine"] for v in vals)/len(vals),speaker_count=len(vals))
            if stage=="asr":
                report["asr_per_sample"]=[{k:v for k,v in row.items() if k!="_normalized"} for row in result["scores"]]
        report["status"]="completed" if all(v=="completed" for v in report["score_stages"].values()) else "partial"
        report["stage"]="finished"
    except Exception as exc:
        report["error_type"]=type(exc).__name__
        import traceback
        (OUT/"private-error.log").write_text(traceback.format_exc(),encoding="utf-8")
        report.setdefault("diagnostics",{})["parent"]=error_signature(OUT/"private-error.log")
    finally:
        report["seconds"]=round(time.monotonic()-start,3)
        Path("/kaggle/working/s1-next-result.json").write_text(json.dumps(report,allow_nan=False,indent=2))
        print("RMS polish status",report["status"],report["stage"],flush=True)


if __name__=="__main__":
    main()
