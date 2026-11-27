"""Expanded S1 alignment pilot; fresh native 48k EN, disjoint KO replay."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path("/kaggle/tmp/GPT-SoVITS")
OUT=Path("/kaggle/working/expanded-alignment-workers")

def main():
    report={"code_sha":REQUEST["code_sha"],"status":"failed","stage":"mount","training_performed":True,"cfm_synthesis_performed":False,"source_run":45948297170}
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    listen=Path("/kaggle/working/expanded-alignment")
    listen.mkdir()
    try:
        def source(kernel,sha,cases,key):
            paths=[p for p in Path("/kaggle/input").rglob("s1-next-result.json") if kernel in p.parts]
            assert len(paths)==1
            r=json.loads(paths[0].read_text())
            assert r["code_sha"]==sha and r["status"]=="completed" and r[key]==cases and all(v==0 for v in r["workers"].values())
            return paths[0].parent
        kernel="gpt-sovits-s1-next-45948297170"
        assert REQUEST["source_kernel"]=="pymood1/"+kernel
        prior=source(kernel,"e4ae8f3bbc974148572a20ee5e4983f4a3b2ac0e",240,"actual_cases")/"training-repair"
        ko=prior/"ko_guard"
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
        setup_start=time.monotonic()
        setup=ROOT/"tools/colab_fast_setup.py"
        original=setup.read_bytes()
        try:
            setup.write_text(original.decode().replace("/content/","/kaggle/tmp/"),encoding="utf-8")
            subprocess.run([sys.executable,"-u",str(setup)],cwd=ROOT,env=env,check=True)
        finally:
            setup.write_bytes(original)
        shutil.rmtree("/kaggle/tmp/gpt_sovits_downloads",ignore_errors=True)
        report["setup_seconds"]=time.monotonic()-setup_start
        gpus=subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True).splitlines()
        assert len(gpus)==2 and all("T4" in x for x in gpus)
        report["gpu_count"]=2
        sys.path[:0]=[str(ROOT),str(ROOT/"GPT_SoVITS")]
        os.chdir(ROOT)
        import hashlib
        pretrained=ROOT/"GPT_SoVITS/pretrained_models"
        report["checkpoint_sha256"]={}
        for relative in ("s1v3.ckpt","gsv-v5-pretrained/s2Gv5dev.pth","gsv-v5-pretrained/vocoder.pth"):
            with (pretrained/relative).open("rb") as f:
                report["checkpoint_sha256"][relative]=hashlib.file_digest(f,"sha256").hexdigest()
        assert report["checkpoint_sha256"]["s1v3.ckpt"]=="87133414860ea14ff6620c483a3db5ed07b44be42e2c3fcdad65523a729a745a"
        report["stage"]="corpus"
        import tools.kaggle_s1_s2_acoustic_cfg as runner
        runner.OUT=OUT
        run_workers=runner.run_workers
        error_signature=runner.error_signature
        runner.ensure_frontend_resources(env)
        from tools.prepare_s1_expanded_vctk import prepare
        from tools.prepare_s1_korean_replay import prepare as prepare_ko
        old=[json.loads((prior/name/"manifest.json").read_text()) for name in ("train","validation","test")]
        revision=old[0]["revision"];assert revision=="0492fa00cf72afb16e753e95d0ec3439eebb07a6"
        blocked=[p for m in old for p in m["pairs"]]
        def items(pairs):return [r for p in pairs for r in (p["reference"],*p["targets"])]
        held=items(blocked)
        pool=Path("/kaggle/tmp/expanded-vctk")
        prepare(pool,24,4,revision=revision,excluded_speakers=[p["speaker_id"] for p in blocked],
                excluded_texts=[r["text"] for r in held],excluded_hashes=[r["sha256"] for r in held],max_shards=32)
        m=json.loads((pool/"manifest.json").read_text());pairs=m["pairs"];assert len(pairs)==24
        import random
        random.Random(20261127).shuffle(pairs)
        for name,pp in (("train",pairs[:16]),("validation",pairs[16:20]),("test",pairs[20:])):
            dest=listen/name;dest.mkdir()
            for r in items(pp):shutil.copyfile(pool/r["path"],dest/r["path"])
            (dest/"manifest.json").write_text(json.dumps(dict(m,pairs=pp),ensure_ascii=False,indent=2))
        km=json.loads((ko/"manifest.json").read_text())
        assert km["sample_rate"]==16000 and len(km["pairs"])==2
        shutil.copytree(ko,listen/"ko_guard")
        prepare_ko(listen/"ko_train",8,3,revision=km["revision"],
            excluded_speakers=[p["speaker_id"] for p in km["pairs"]],
            excluded_texts=[r["text"] for r in items(km["pairs"])],
            excluded_hashes=[r["sha256"] for r in items(km["pairs"])])
        report["corpus"]={"train_speakers":16,"validation_speakers":4,"test_speakers":4,"ko_train_speakers":8,"ko_guard_speakers":2,
                           "en_sample_rate":48000,"ko_sample_rate":16000,"en_test_excludes_v16_speakers_text_pcm":True,
                           "pretraining_overlap":"unknown","en_revision":revision,"ko_revision":km["revision"]}
        common=[sys.executable,"-u",str(ROOT/"tools/s1_expanded_alignment_pilot.py"),
                "--train",str(listen/"train"),"--validation",str(listen/"validation"),"--test",str(listen/"test"),
                "--ko",str(listen/"ko_guard"),"--ko-train",str(listen/"ko_train"),"--steps","240"]
        cmds=[common+["--output",str(OUT/f"gpu{i}"),"--seed",str(20261127+i)] for i in range(2)]
        report["stage"]="expanded_alignment"
        run_workers(cmds,[dict(env,CUDA_VISIBLE_DEVICES=str(i)) for i in range(2)],["eval0","eval1"],report)
        reps=[json.loads((OUT/f"gpu{i}"/"metrics.json").read_text()) for i in range(2)]
        assert all(r["status"]=="completed" and r["updates_per_arm"]==240 and len(r["rows"])==216 for r in reps)
        report["actual_cases"]=sum(len(r["rows"]) for r in reps)
        report["expected_cases"]=432
        assert report["actual_cases"]==432
        report["replications"]=reps
        for i in range(2):
            for name in reps[i]["arms"]:
                p=OUT/f"gpu{i}"/(name+".pt");assert p.is_file() and p.stat().st_size>0
        report["production_changed"]=False
        report["dpo_enabled"]=False
        report["waveform_evaluation"]=False
        report["status"]="completed"
        report["stage"]="finished"
    except Exception as exc:
        report["error_type"]=type(exc).__name__
        import traceback
        (OUT/"private-error.log").write_text(traceback.format_exc(),encoding="utf-8")
        if "error_signature" in locals(): report.setdefault("diagnostics",{})["parent"]=error_signature(OUT/"private-error.log")
    finally:
        report["seconds"]=round(time.monotonic()-start,3)
        Path("/kaggle/working/s1-next-result.json").write_text(json.dumps(report,allow_nan=False,indent=2))
        print("Expanded alignment status",report["status"],report.get("actual_cases",0),flush=True)

if __name__=="__main__":
    main()
