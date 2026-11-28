"""Codebook-aware gated token restoration; independent validation and test."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path("/kaggle/tmp/GPT-SoVITS")
OUT=Path("/kaggle/working/token-aware-workers")

def main():
    report={"code_sha":REQUEST["code_sha"],"status":"failed","stage":"mount","training_performed":False,"cfm_synthesis_performed":False,"source_run":45923560566}
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    listen=Path("/kaggle/working/token-aware")
    listen.mkdir()
    try:
        def source(kernel,sha,cases,key):
            paths=[p for p in Path("/kaggle/input").rglob("s1-next-result.json") if kernel in p.parts]
            assert len(paths)==1
            r=json.loads(paths[0].read_text())
            assert r["code_sha"]==sha and r["status"]=="completed" and r[key]==cases and all(v==0 for v in r["workers"].values())
            return paths[0].parent
        kernel="gpt-sovits-s1-next-45923560566"
        assert REQUEST["source_kernel"]=="pymood1/"+kernel
        prior=source(kernel,"f4ff7ade3263330d7f993a66c25110023e9f6f8d",192,"actual_cases")/"semantic-robustness"
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
        train=prior/"train";validation=prior/"test"
        tr=json.loads((train/"manifest.json").read_text())
        va=json.loads((validation/"manifest.json").read_text())
        revision=tr["revision"]
        assert revision==va["revision"]=="0492fa00cf72afb16e753e95d0ec3439eebb07a6"
        test=Path("/kaggle/tmp/token-aware-test")
        blocked=tr["pairs"]+va["pairs"]
        held=[i for p in blocked for i in [p["reference"],*p["targets"]]]
        prepare(test,4,2,revision=revision,excluded_speakers=[p["speaker_id"] for p in blocked],
                excluded_texts=[i["text"] for i in held],excluded_hashes=[i["sha256"] for i in held])
        tm=json.loads((test/"manifest.json").read_text())
        assert len(tr["pairs"])==6 and len(va["pairs"])==4 and len(tm["pairs"])==4
        report["corpus"]={"dataset":tm["dataset"],"revision":revision,"sample_rate":48000,"train_speakers":6,"validation_speakers":4,"test_speakers":4,"license":tm["license"],"pretraining_overlap":"unknown"}
        for name,folder in (("train",train),("validation",validation),("test",test)):
            shutil.copytree(folder,listen/name)
        report["stage"]="token_aware"
        common=[sys.executable,"-u",str(ROOT/"tools/s1_token_aware_restoration.py"),"--train",str(train),"--validation",str(validation),"--test",str(test)]
        cmds=[common+["--output",str(OUT/f"gpu{i}"),"--seed",str(20261127+i)] for i in range(2)]
        run_workers(cmds,[dict(env,CUDA_VISIBLE_DEVICES=str(i)) for i in range(2)],["eval0","eval1"],report)
        report["training_performed"]=True
        reps=[json.loads((OUT/f"gpu{i}"/"metrics.json").read_text()) for i in range(2)]
        report["replications"]=reps
        report["actual_cases"]=sum(len(r["rows"]) for r in reps)
        report["expected_cases"]=288
        assert report["actual_cases"]==288
        assessment=[]
        for r in reps:
            assert r["status"]=="completed" and r["training"]["updates"]==720
            s=r["summary"]
            gate={"clean_drift_le_2pct":s["clean-gated"]["ued"]<=.02,
                  "seen_noise_improves":all(s[k+"-gated"]["ued"]<s[k+"-baseline"]["ued"] for k in ("noise10","band8")),
                  "unseen_room_nonregression":s["room-gated"]["ued"]<=s["room-baseline"]["ued"],
                  "no_large_perplexity_drop":all(s[k+"-gated"]["mean_perplexity"]>=.8*s[k+"-baseline"]["mean_perplexity"] for k in ("clean","noise10","band8","room")),
                  "no_large_repeat_growth":all(s[k+"-gated"]["max_repeat"]<=s[k+"-baseline"]["max_repeat"]+4 for k in ("clean","noise10","band8","room"))}
            assessment.append({"seed":r["seed"],"gate":gate,"passed":all(gate.values())})
        report["assessment"]={"replications":assessment,"pass_both":all(r["passed"] for r in assessment)}
        for i in range(2):
            shutil.copyfile(OUT/f"gpu{i}"/"token-aware.pth",listen/f"token-aware-seed-{20261127+i}.pth")
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
        print("Token aware status",report["status"],report.get("actual_cases",0),flush=True)

if __name__=="__main__":
    main()
