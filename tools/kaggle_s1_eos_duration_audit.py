"""S1 EOS and duration audit with paired downstream synthesis."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path("/kaggle/tmp/GPT-SoVITS")
OUT=Path("/kaggle/working/eos-workers")

def main():
    report={"code_sha":REQUEST["code_sha"],"status":"failed","stage":"mount","training_performed":False,"cfm_synthesis_performed":True,"source_run":45925702174}
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    listen=Path("/kaggle/working/eos-audit")
    listen.mkdir()
    try:
        def source(kernel,sha,cases,key):
            paths=[p for p in Path("/kaggle/input").rglob("s1-next-result.json") if kernel in p.parts]
            assert len(paths)==1
            r=json.loads(paths[0].read_text())
            assert r["code_sha"]==sha and r["status"]=="completed" and r[key]==cases and all(v==0 for v in r["workers"].values())
            return paths[0].parent
        kernel="gpt-sovits-s1-next-45925702174"
        assert REQUEST["source_kernel"]=="pymood1/"+kernel
        prior=source(kernel,"d3cf5d73274e4b716e8a671da674a248a5b82c21",576,"actual_cases")/"multi-restoration"
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
        from tools.prepare_acoustic_vctk_pairs import prepare
        train=prior/"train";validation=prior/"test"
        tr=json.loads((train/"manifest.json").read_text())
        va=json.loads((validation/"manifest.json").read_text())
        revision=tr["revision"]
        assert revision==va["revision"]=="0492fa00cf72afb16e753e95d0ec3439eebb07a6"
        test=Path("/kaggle/tmp/eos-test")
        oldval=json.loads((prior/"validation/manifest.json").read_text())
        blocked=tr["pairs"]+va["pairs"]+oldval["pairs"]
        held=[i for p in blocked for i in [p["reference"],*p["targets"]]]
        prepare(test,4,2,revision=revision,excluded_speakers=[p["speaker_id"] for p in blocked],
                excluded_texts=[i["text"] for i in held],excluded_hashes=[i["sha256"] for i in held])
        tm=json.loads((test/"manifest.json").read_text())
        assert len(tr["pairs"])==6 and len(va["pairs"])==4 and len(tm["pairs"])==4
        report["corpus"]={"dataset":tm["dataset"],"revision":revision,"sample_rate":48000,"train_speakers":6,"validation_speakers":4,"test_speakers":4,"license":tm["license"],"pretraining_overlap":"unknown"}
        for name,folder in (("train",train),("validation",validation),("test",test)):
            shutil.copytree(folder,listen/name)
        report["stage"]="eos_audit"
        common=[sys.executable,"-u",str(ROOT/"tools/s1_eos_duration_audit.py"),"--train",str(train),"--validation",str(validation),"--test",str(test),"--synthesize"]
        cmds=[common+["--output",str(OUT/f"gpu{i}"),"--seed",str(20261127+i)] for i in range(2)]
        run_workers(cmds,[dict(env,CUDA_VISIBLE_DEVICES=str(i)) for i in range(2)],["eval0","eval1"],report)
        report["duration_fit_performed"]=True
        reps=[json.loads((OUT/f"gpu{i}"/"metrics.json").read_text()) for i in range(2)]
        report["replications"]=reps
        report["actual_cases"]=sum(len(r["rows"]) for r in reps)
        report["expected_cases"]=48
        assert report["actual_cases"]==48 and all(len(r["forced"])==8 for r in reps)
        report["stage"]="scoring"
        import site
        libs=[str(p) for folder in site.getsitepackages() for p in sorted(Path(folder).glob("nvidia/*/lib")) if p.is_dir()]
        libs.extend(str(p) for p in (Path("/usr/local/cuda/lib64"),Path("/usr/lib/x86_64-linux-gnu")) if p.is_dir())
        env["LD_LIBRARY_PATH"]=os.pathsep.join(dict.fromkeys(libs+[p for p in env.get("LD_LIBRARY_PATH","").split(os.pathsep) if p]))
        req={"texts":[t["text"] for p in tm["pairs"] for t in p["targets"]],
             "references":[str(test/p["reference"]["path"]) for p in tm["pairs"]],
             "samples":[],"cpu_threads":2,"language":"en"}
        for i in range(2):
            for r in reps[i]["rows"]:
                ri,ti,st=r["speaker"],r["target"],int(r["strength"])
                wav=OUT/f"gpu{i}"/f"speaker-{ri:02}-target-{ti:02}-bias-{st}.wav"
                assert wav.is_file() and wav.stat().st_size>44
                dest=listen/f"seed-{i}-speaker-{ri:02}-target-{ti:02}-bias-{st}.wav"
                shutil.copyfile(wav,dest)
                req["samples"].append(dict(label=f"seed{i}-bias{st}",reference_index=ri,text_index=(ri-1)*2+ti,path=str(dest)))
        path=OUT/"scoring.json";path.write_text(json.dumps(req,ensure_ascii=False))
        from tools.asr.fasterwhisper_asr import download_model
        download_model("large-v3")
        asr=OUT/"asr.json";sv=OUT/"sv.json"
        cmds=[["timeout","1200",sys.executable,"-u",str(ROOT/"tools/score_s1_fp32_asr.py"),"--source",str(path),"--output",str(asr),"--backend","ct2","--expected-count","48"],
              ["timeout","900",sys.executable,"-u",str(ROOT/"tools/score_s1_stack_waveforms.py"),str(path),"speaker",str(sv)]]
        run_workers(cmds,[dict(env,CUDA_VISIBLE_DEVICES=str(i)) for i in range(2)],["asr-fp32","speaker"],report)
        aa,bb=json.loads(asr.read_text()),json.loads(sv.read_text())
        assert aa["backend"]=="ct2_cuda_float32" and bb["backend"]=="cuda_fp32"
        assert len(aa["scores"])==len(bb["scores"])==48
        report["score_backends"]={"asr":aa["backend"],"speaker":bb["backend"]}
        report["waveform_summary"]={}
        for i in range(2):
            for st in (0,2,4):
                label=f"seed{i}-bias{st}"
                rr=[(ar,sr) for q,ar,sr in zip(req["samples"],aa["scores"],bb["scores"]) if q["label"]==label]
                assert len(rr)==8 and all(ar["label"]==label for ar,sr in rr)
                report["waveform_summary"][label]={"cer":sum(ar["char_errors"] for ar,sr in rr)/sum(ar["char_count"] for ar,sr in rr),
                    "speaker_cosine":sum(sr["speaker_cosine"] for ar,sr in rr)/len(rr)}
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
        print("EOS audit status",report["status"],report.get("actual_cases",0),flush=True)

if __name__=="__main__":
    main()
