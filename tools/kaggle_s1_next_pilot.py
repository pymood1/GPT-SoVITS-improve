"""Private Kaggle pilot launcher, public corpus only; REQUEST is injected by CI."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path("/kaggle/tmp/GPT-SoVITS")
OUT=Path("/kaggle/working/s1-next-pilot")
REPO="https://github.com/pymood1/GPT-SoVITS-improve.git"


def main():
    sha=REQUEST["code_sha"]
    report={"code_sha":sha,"status":"failed","stage":"clone"}
    started=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    try:
        ROOT.mkdir(parents=True,exist_ok=False)
        subprocess.run(["git","init",str(ROOT)],check=True)
        subprocess.run(["git","-C",str(ROOT),"fetch","--depth","1",REPO,sha],check=True)
        subprocess.run(["git","-C",str(ROOT),"checkout","--detach","FETCH_HEAD"],check=True)
        if subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()!=sha:
            raise ValueError("Pinned commit mismatch")
        report["stage"]="runtime_setup"
        env=os.environ.copy()
        env.update(HF_HOME="/kaggle/tmp/huggingface",TMPDIR="/kaggle/tmp",
                   PYTHONUNBUFFERED="1",PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/"GPT_SoVITS")]))
        setup=ROOT/"tools/colab_fast_setup.py"
        original=setup.read_bytes()
        try:
            # Shared installer used previously on Kaggle; no Colab runtime.
            setup.write_text(original.decode().replace("/content/","/kaggle/tmp/"),encoding="utf-8")
            subprocess.run([sys.executable,"-u",str(setup)],cwd=ROOT,env=env,check=True)
        finally:
            setup.write_bytes(original)
        shutil.rmtree("/kaggle/tmp/gpt_sovits_downloads",ignore_errors=True)
        report["stage"]="gpu_check"
        gpus=subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True).splitlines()
        report["gpu_count"]=len(gpus)
        if len(gpus)!=2 or any("T4" not in g for g in gpus):
            raise RuntimeError("Expected T4 x2")
        teacher=ROOT/"GPT_SoVITS/pretrained_models/s1v3.ckpt"
        s2=ROOT/"GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"
        corpus=Path("/kaggle/tmp/s1-next-fleurs")
        report["stage"]="public_corpus"
        subprocess.run([sys.executable,"-u",str(ROOT/"tools/prepare_s1_next_fleurs.py"),
                        "--root",str(corpus),"--teacher",str(teacher),"--train-count","64",
                        "--val-count","16"],cwd=ROOT,env=env,check=True)
        # Shared public corpus; two independent GPU workers. No personal dataset mount.
        report["stage"]="parallel_pilots"
        common=["--teacher",str(teacher),"--train",str(corpus/"train"),
                "--validation",str(corpus/"validation")]
        train_env=dict(env,CUDA_VISIBLE_DEVICES="0")
        bridge_env=dict(env,CUDA_VISIBLE_DEVICES="1")
        train=[sys.executable,"-u",str(ROOT/"tools/train_s1_next.py"),*common,
               "--output",str(OUT/"s1"),"--epochs","3","--max-steps","40",
               "--max-tokens","40000","--tokens-per-update","1024","--warmup-steps","5",
               "--teacher-cache","/kaggle/tmp/s1-next-teacher-cache"]
        provenance=json.loads((corpus/"train/provenance.json").read_text())
        if not provenance["multilingual_bert"]:
            train.append("--allow-missing-bert")
        bridge=[sys.executable,"-u",str(ROOT/"tools/train_s1_next_bridge_pilot.py"),*common,
                "--s2",str(s2),"--output",str(OUT/"bridge")]
        with (OUT/"s1.log").open("w") as l1,(OUT/"bridge.log").open("w") as l2:
            p1=subprocess.Popen(train,cwd=ROOT,env=train_env,stdout=l1,stderr=subprocess.STDOUT)
            p2=subprocess.Popen(bridge,cwd=ROOT,env=bridge_env,stdout=l2,stderr=subprocess.STDOUT)
            try:
                while p1.poll() is None or p2.poll() is None:
                    print("[pilot] S1",p1.poll(),"bridge",p2.poll(),flush=True)
                    time.sleep(15)
            finally:
                for p in (p1,p2):
                    if p.poll() is None:
                        p.terminate()
                        p.wait()
        report["s1_exit"]=p1.returncode
        report["bridge_exit"]=p2.returncode
        for name in ("s1","bridge"):
            f=OUT/name/"metrics.json"
            if f.is_file():
                report[name]=json.loads(f.read_text())
        if p1.returncode==p2.returncode==0 and "s1" in report and "bridge" in report:
            report.update(status="completed",stage="finished")
        else:
            report["status"]="partial" if "s1" in report or "bridge" in report else "failed"
    except Exception as exc:
        report["error_type"]=type(exc).__name__
        import traceback
        (OUT/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
    finally:
        report["seconds"]=round(time.monotonic()-started,3)
        (Path("/kaggle/working")/"s1-next-result.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
        print("Pilot status",report["status"],report["stage"],flush=True)

if __name__=="__main__":
    main()
