"""Semantic robustness: two independent FP32 feature adapter fits."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path("/kaggle/tmp/GPT-SoVITS")
OUT=Path("/kaggle/working/semantic-workers")

def main():
    report={"code_sha":REQUEST["code_sha"],"status":"failed","stage":"mount","training_performed":False,"cfm_synthesis_performed":False,"source_run":45921918936}
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    listen=Path("/kaggle/working/semantic-robustness")
    listen.mkdir()
    try:
        def source(kernel,sha,cases,key):
            paths=[p for p in Path("/kaggle/input").rglob("s1-next-result.json") if kernel in p.parts]
            assert len(paths)==1
            r=json.loads(paths[0].read_text())
            assert r["code_sha"]==sha and r["status"]=="completed" and r[key]==cases and all(v==0 for v in r["workers"].values())
            return paths[0].parent
        kernel="gpt-sovits-s1-next-45921918936"
        assert REQUEST["source_kernel"]=="pymood1/"+kernel
        prior=source(kernel,"b42878d19ff8fb92ee5f923d5852e7dcdd638b42",32,"actual_cases")/"acoustic-curriculum"
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
        report["stage"]="control_assets"
        from huggingface_hub import HfApi,hf_hub_download
        remote="lj1995/GPT-SoVITS"
        revision=HfApi().model_info(remote).sha
        report["control_asset_revision"]=revision
        pretrained=ROOT/"GPT_SoVITS/pretrained_models"
        for relative in ("v2Pro/s2Gv2ProPlus.pth","sv/pretrained_eres2netv2w24s4ep4.ckpt"):
            cached=hf_hub_download(remote,relative,revision=revision)
            dest=pretrained/relative
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(cached,dest)
        import hashlib
        report["checkpoint_sha256"]={}
        for relative in ("s1v3.ckpt","v2Pro/s2Gv2ProPlus.pth","gsv-v5-pretrained/s2Gv5dev.pth","gsv-v5-pretrained/vocoder.pth"):
            with (pretrained/relative).open("rb") as f:
                report["checkpoint_sha256"][relative]=hashlib.file_digest(f,"sha256").hexdigest()
        report["stage"]="corpus"
        import tools.kaggle_s1_s2_acoustic_cfg as runner
        runner.OUT=OUT
        run_workers=runner.run_workers
        error_signature=runner.error_signature
        from tools.prepare_acoustic_vctk_pairs import prepare
        old=json.loads((prior/"manifest.json").read_text())
        revision=old["revision"]
        assert revision=="0492fa00cf72afb16e753e95d0ec3439eebb07a6"
        test=Path("/kaggle/tmp/semantic-test")
        train=Path("/kaggle/tmp/semantic-train")
        prepare(test,4,2,revision=revision)
        tm=json.loads((test/"manifest.json").read_text())
        held=[i for p in tm["pairs"] for i in [p["reference"],*p["targets"]]]
        prepare(train,6,2,revision=revision,excluded_speakers=[p["speaker_id"] for p in tm["pairs"]],
                excluded_texts=[i["text"] for i in held],excluded_hashes=[i["sha256"] for i in held])
        tr=json.loads((train/"manifest.json").read_text())
        assert len(tr["pairs"])==6 and len(tm["pairs"])==4
        assert not {p["speaker_id"] for p in tr["pairs"]}&{p["speaker_id"] for p in tm["pairs"]}
        report["corpus"]={"dataset":tm["dataset"],"revision":revision,"sample_rate":48000,"train_speakers":6,"test_speakers":4,"license":tm["license"],"pretraining_overlap":"unknown"}
        for name,folder in (("train",train),("test",test)):
            dest=listen/name;shutil.copytree(folder,dest)
        report["stage"]="semantic_adapter"
        common=[sys.executable,"-u",str(ROOT/"tools/s1_semantic_robustness.py"),"--train",str(train),"--test",str(test)]
        cmds=[common+["--output",str(OUT/f"gpu{i}"),"--seed",str(20261127+i)] for i in range(2)]
        run_workers(cmds,[dict(env,CUDA_VISIBLE_DEVICES=str(i)) for i in range(2)],["eval0","eval1"],report)
        report["training_performed"]=True
        report["replications"]=[json.loads((OUT/f"gpu{i}"/"metrics.json").read_text()) for i in range(2)]
        from tools.s1_semantic_robustness_gate import assess
        report["assessment"]=assess(report["replications"])
        report["actual_cases"]=sum(len(r["rows"]) for r in report["replications"])
        report["expected_cases"]=192
        assert report["actual_cases"]==192
        for i in range(2):
            shutil.copyfile(OUT/f"gpu{i}"/"ssl-adapter.pth",listen/f"ssl-adapter-seed-{20261127+i}.pth")
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
        print("Semantic robustness status",report["status"],report.get("actual_cases",0),flush=True)

if __name__=="__main__":
    main()
