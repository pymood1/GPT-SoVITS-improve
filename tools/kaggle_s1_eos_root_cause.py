"""Root cause audit on matched EN/KO audio and two official quantizers."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path("/kaggle/tmp/GPT-SoVITS")
OUT=Path("/kaggle/working/root-cause-workers")

def main():
    report={"code_sha":REQUEST["code_sha"],"status":"failed","stage":"mount","training_performed":False,"cfm_synthesis_performed":False,"source_run":45940943141}
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    listen=Path("/kaggle/working/eos-root-cause")
    listen.mkdir()
    try:
        def source(kernel,sha,cases,key):
            paths=[p for p in Path("/kaggle/input").rglob("s1-next-result.json") if kernel in p.parts]
            assert len(paths)==1
            r=json.loads(paths[0].read_text())
            assert r["code_sha"]==sha and r["status"]=="completed" and r[key]==cases and all(v==0 for v in r["workers"].values())
            return paths[0].parent
        kernel="gpt-sovits-s1-next-45940943141"
        assert REQUEST["source_kernel"]=="pymood1/"+kernel
        en=source(kernel,"41d6bcaf4d76299e7906cadd34c3bd607e4c9e95",48,"actual_cases")/"eos-audit/test"
        ko=source("gpt-sovits-s1-next-45915504105","125b2ec239356b01cf78db15ada90079330fd0f8",32,"actual_cases")/"acoustic-paired"
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
        runner.ensure_frontend_resources(env)
        em=json.loads((en/"manifest.json").read_text());km=json.loads((ko/"manifest.json").read_text())
        assert len(em["pairs"])==4 and len(km["pairs"])==2
        assert em["sample_rate"]==48000 and km["sample_rate"]==16000
        report["corpus"]={"en":{"speakers":4,"sample_rate":48000,"revision":em["revision"]},
                           "ko":{"speakers":2,"sample_rate":16000,"revision":km["revision"]}}
        subprocess.run([sys.executable,"-u",str(ROOT/"tools/audit_s1_training_contracts.py")],cwd=ROOT,env=env,check=True)
        common=[sys.executable,"-u",str(ROOT/"tools/s1_eos_root_cause.py"),"--en",str(en),"--ko",str(ko)]
        versions=("v5dev","v2ProPlus")
        cmds=[common+["--output",str(OUT/f"gpu{i}"),"--version",version] for i,version in enumerate(versions)]
        report["stage"]="root_cause"
        run_workers(cmds,[dict(env,CUDA_VISIBLE_DEVICES=str(i)) for i in range(2)],["eval0","eval1"],report)
        reps=[json.loads((OUT/f"gpu{i}"/"metrics.json").read_text()) for i in range(2)]
        assert reps[0]["s1_sha"]==reps[1]["s1_sha"]
        assert all(len(r["rows"])==72 and len(r["generation"])==12 for r in reps)
        report["actual_cases"]=sum(len(r["rows"])+len(r["generation"]) for r in reps)
        report["expected_cases"]=168
        assert report["actual_cases"]==168
        from tools.s1_semantic_robustness import token_metrics
        assert set(reps[0]["tokens"])==set(reps[1]["tokens"])
        report["cross_version_tokens"]=[]
        for k in sorted(reps[0]["tokens"]):
            for field in ("target","train_target","plain","native","pad03","train_ref"):
                aa=reps[0]["tokens"][k][field];bb=reps[1]["tokens"][k][field]
                report["cross_version_tokens"].append(dict(case=k,field=field,**token_metrics(aa,bb)))
        for r in reps:
            r["preprocessing_tokens"]=[]
            for k,t in r.pop("tokens").items():
                r["preprocessing_tokens"].append(dict(case=k,target_raw_vs_train=token_metrics(t["target"],t["train_target"]),
                    ref_raw_vs_train=token_metrics(t["plain"],t["train_ref"]),
                    ref_native_vs_300ms=token_metrics(t["native"],t["pad03"])))
        report["replications"]=reps
        import statistics as st
        report["summary"]={}
        for r in reps:
            report["summary"][r["version"]]={}
            for lang in ("en","all_ko"):
                for arm in ("standalone","native_prompt","no_padding","exact_300ms","train_standalone","train_concat"):
                    rr=[x for x in r["rows"] if x["language"]==lang and x["arm"]==arm]
                    assert len(rr)==(8 if lang=="en" else 4)
                    report["summary"][r["version"]][lang+"-"+arm]=dict(cases=len(rr),
                        median_eos_prob=st.median(x["eos_prob"] for x in rr),
                        median_eos_rank=st.median(x["eos_rank"] for x in rr),
                        mean_nll=st.mean(x["nll"] for x in rr),
                        mean_top1=st.mean(x["token_top1"] for x in rr),
                        max_native_last_diff=max(x["native_last_max_abs"] for x in rr))
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
        print("Root cause audit status",report["status"],report.get("actual_cases",0),flush=True)

if __name__=="__main__":
    main()
