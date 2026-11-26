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


def mounted_checkpoint(kernel):
    slug=kernel.split("/")[-1]
    if kernel!="pymood1/gpt-sovits-s1-next-45739171412":
        raise ValueError("Unexpected source kernel")
    root=Path("/kaggle/input")
    matches=[p for p in root.rglob("s1-next-best.pt") if slug in p.parts and p.parent.name=="kd"]
    if len(matches)!=1:
        raise FileNotFoundError("Expected one pinned prior KD checkpoint")
    checkpoint=matches[0]
    results=[p for p in root.rglob("s1-next-result.json") if slug in p.parts]
    if len(results)!=1:
        raise FileNotFoundError("Expected one pinned prior result")
    previous=json.loads(results[0].read_text())
    if previous.get("code_sha")!="604b036706e94559aca68b893fcdbe1340ae28f6" or previous.get("status")!="completed":
        raise ValueError("Prior experiment result mismatch")
    if checkpoint.stat().st_size<1000000:
        raise ValueError("Checkpoint truncated")
    return checkpoint


def main():
    sha=REQUEST["code_sha"]
    report={"code_sha":sha,"status":"failed","stage":"clone"}
    started=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    try:
        report["stage"]="prior_checkpoint_mount"
        checkpoint=mounted_checkpoint(REQUEST["source_kernel"])
        report["stage"]="clone"
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
                        "--root",str(corpus),"--teacher",str(teacher),"--train-count","384",
                        "--val-count","32","--test-count","24"],cwd=ROOT,env=env,check=True)
        # Shared public corpus; two independent GPU workers. No personal dataset mount.
        report["stage"]="parallel_pilots"
        common=["--teacher",str(teacher),"--train",str(corpus/"train"),
                "--validation",str(corpus/"validation"),"--test",str(corpus/"test")]
        train_env=dict(env,CUDA_VISIBLE_DEVICES="0")
        bridge_env=dict(env,CUDA_VISIBLE_DEVICES="1")
        provenance=json.loads((corpus/"train/provenance.json").read_text())
        train=[sys.executable,"-u",str(ROOT/"tools/train_s1_next.py"),*common,
               "--epochs","8","--max-steps","600","--max-tokens","600000",
               "--init-checkpoint",str(checkpoint),
               "--tokens-per-update","1024","--warmup-steps","20",
               "--rollout-count","6","--teacher-cache","/kaggle/tmp/s1-next-teacher-cache"]
        if not provenance["multilingual_bert"]:
            train.append("--allow-missing-bert")
        bridge=[sys.executable,"-u",str(ROOT/"tools/train_s1_next_bridge_pilot.py"),*common,
                "--s2",str(s2),"--output",str(OUT/"bridge"),"--train-count","32",
                "--val-count","8","--test-count","12","--epochs","4","--compare-conditioning"]
        report["s1"]={}
        report["s1_exit"]={}
        with (OUT/"bridge.log").open("w") as log:
            p2=subprocess.Popen(bridge,cwd=ROOT,env=bridge_env,stdout=log,stderr=subprocess.STDOUT)
            try:
                for name,weight in (("kd_continue","0.0"),("trajectory_kd","0.5")):
                    with (OUT/(name+".log")).open("w") as l1:
                        extra=["--teacher-rollout"] if name=="kd_continue" else []
                        p1=subprocess.Popen(train+["--output",str(OUT/name),"--alpha","0.5","--trajectory-weight",weight]+extra,
                                            cwd=ROOT,env=train_env,stdout=l1,stderr=subprocess.STDOUT)
                        try:
                            while p1.poll() is None:
                                print("[pilot] S1",name,"bridge",p2.poll(),flush=True)
                                time.sleep(15)
                        finally:
                            if p1.poll() is None:
                                p1.terminate()
                                p1.wait()
                    report["s1_exit"][name]=p1.returncode
                    f=OUT/name/"metrics.json"
                    if f.is_file():
                        report["s1"][name]=json.loads(f.read_text())
                while p2.poll() is None:
                    print("[pilot] S1 arms finished; bridge running",flush=True)
                    time.sleep(15)
            finally:
                if p2.poll() is None:
                    p2.terminate()
                    p2.wait()
        report["bridge_exit"]=p2.returncode
        f=OUT/"bridge/metrics.json"
        if f.is_file():
            report["bridge"]=json.loads(f.read_text())
        if all(report["s1_exit"].get(n)==0 and n in report["s1"] for n in ("kd_continue","trajectory_kd")) and p2.returncode==0 and "bridge" in report:
            report.update(status="completed",stage="finished")
        else:
            report["status"]="partial" if report["s1"] or "bridge" in report else "failed"
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
