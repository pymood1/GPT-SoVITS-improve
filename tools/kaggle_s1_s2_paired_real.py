"""Same-speaker real-target fixed-token S2 factorial, FP32 scoring."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path("/kaggle/tmp/GPT-SoVITS")
OUT=Path("/kaggle/working/paired-workers")

def main():
    report={"code_sha":REQUEST["code_sha"],"status":"failed","stage":"mount","training_performed":False,"cfm_synthesis_performed":True,"source_run":45913817545}
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    listen=Path("/kaggle/working/acoustic-paired")
    listen.mkdir()
    try:
        def source(kernel,sha,cases,key):
            paths=[p for p in Path("/kaggle/input").rglob("s1-next-result.json") if kernel in p.parts]
            assert len(paths)==1
            r=json.loads(paths[0].read_text())
            assert r["code_sha"]==sha and r["status"]=="completed" and r[key]==cases
            return paths[0].parent
        kernel="gpt-sovits-s1-next-45913817545"
        assert REQUEST["source_kernel"]=="pymood1/"+kernel
        source(kernel,"ee5f9faeef0589e7e2820e60a71e85b98042c1f4",16,"actual_cases")
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
        report["stage"]="paired_corpus"
        import tools.kaggle_s1_s2_acoustic_cfg as runner
        runner.OUT=OUT
        run_workers=runner.run_workers
        error_signature=runner.error_signature
        runner.ensure_frontend_resources(env)
        from tools.prepare_acoustic_zeroth_pairs import prepare
        corpus=Path("/kaggle/tmp/paired-corpus")
        prepare(corpus,2,2)
        manifest=json.loads((corpus/"manifest.json").read_text())
        assert manifest["verified_same_speaker"] and len(manifest["pairs"])==2
        report["corpus"]={k:manifest[k] for k in ("dataset","revision","split","license","sample_rate","verified_same_speaker")}
        report["stage"]="fixed_token_s2"
        (listen/"README.txt").write_text("Two same-speaker reference/target pairs per speaker. Raw target and prompt semantic tokens frozen across factorial. Oracle and true Mel controls have target access and are diagnostic only. V2 native separate control. No training. All spectral comparisons use 7600 Hz bandwidth.")
        common=[sys.executable,"-u",str(ROOT/"tools/s1_s2_paired_real.py"),"--source",str(corpus)]
        cmds=[common+["--output",str(OUT/f"gpu{i}"),"--worker",str(i)] for i in range(2)]
        run_workers(cmds,[dict(env,CUDA_VISIBLE_DEVICES=str(i)) for i in range(2)],["eval0","eval1"],report)
        report["rows"]=[]
        for i in range(2):
            folder=OUT/f"gpu{i}"
            report["rows"].extend(json.loads((folder/"metrics.json").read_text()))
            for p in folder.glob("*.wav"):shutil.copyfile(p,listen/p.name)
            for p in folder.glob("*.npz"):shutil.copyfile(p,listen/p.name)
        report["expected_cases"]=32
        report["actual_cases"]=len(report["rows"])
        assert len(report["rows"])==32
        for p in corpus.glob("*.wav"):shutil.copyfile(p,listen/p.name)
        shutil.copyfile(corpus/"manifest.json",listen/"manifest.json")
        report["stage"]="scoring"
        import site
        libs=[str(p) for folder in site.getsitepackages() for p in sorted(Path(folder).glob("nvidia/*/lib")) if p.is_dir()]
        libs.extend(str(p) for p in (Path("/usr/local/cuda/lib64"),Path("/usr/lib/x86_64-linux-gnu")) if p.is_dir())
        env["LD_LIBRARY_PATH"]=os.pathsep.join(dict.fromkeys(libs+[p for p in env.get("LD_LIBRARY_PATH","").split(os.pathsep) if p]))
        texts=[t["text"] for pair in manifest["pairs"] for t in pair["targets"]]
        arms=("raw","raw-hard","gain","gain-hard","real","true-mel-vocoder","oracle-s2","v2ProPlus")
        request={"texts":texts,"references":[],"samples":[],"cpu_threads":2}
        for ri,pair in enumerate(manifest["pairs"],1):
            request["references"].append(str(listen/pair["reference"]["path"]))
            for ti in range((ri-1)*2+1,ri*2+1):
                for arm in arms:
                    path=listen/f"ref{ri:02}-text{ti:02}-{arm}-level.wav"
                    assert path.is_file() and path.stat().st_size>44
                    request["samples"].append({"label":arm,"reference_index":ri,"text_index":ti,"path":str(path)})
        req=OUT/"scoring-input.json"
        req.write_text(json.dumps(request,ensure_ascii=False))
        from tools.asr.fasterwhisper_asr import download_model
        download_model("large-v3")
        asr=OUT/"asr-fp32.json";sv=OUT/"speaker-scores.json"
        cmds=[["timeout","600",sys.executable,"-u",str(ROOT/"tools/score_s1_fp32_asr.py"),"--source",str(req),"--output",str(asr),"--backend","ct2","--expected-count","32"],
              ["timeout","420",sys.executable,"-u",str(ROOT/"tools/score_s1_stack_waveforms.py"),str(req),"speaker",str(sv)]]
        run_workers(cmds,[dict(env,CUDA_VISIBLE_DEVICES=str(i)) for i in range(2)],["asr-fp32","speaker"],report)
        a,b=json.loads(asr.read_text()),json.loads(sv.read_text())
        assert a["backend"]=="ct2_cuda_float32" and b["backend"]=="cuda_fp32"
        assert len(a["scores"])==len(b["scores"])==32
        report["score_backends"]={"asr":a["backend"],"speaker":b["backend"]}
        indexed={(r["reference_index"],r["text_index"],r["arm"]):r for r in report["rows"]}
        assert len(indexed)==32
        for sample,ar,sr in zip(request["samples"],a["scores"],b["scores"]):
            assert sample["label"]==ar["label"] and sample["reference_index"]==ar["reference_index"] and sample["text_index"]==ar["text_index"]
            row=indexed[sample["reference_index"],sample["text_index"],sample["label"]]
            row.update(char_errors=ar["char_errors"],char_count=ar["char_count"],speaker_cosine=sr["speaker_cosine"])
        report["summary"]={}
        for arm in arms:
            rows=[r for r in report["rows"] if r["arm"]==arm]
            report["summary"][arm]={"cer":sum(r["char_errors"] for r in rows)/sum(r["char_count"] for r in rows),
                                     "speaker_cosine":sum(r["speaker_cosine"] for r in rows)/len(rows)}
        report["stage"]="spectral"
        from tools.acoustic_spectral_compare import features,compare
        import soundfile as sf
        for ri in (1,2):
            for ti in range((ri-1)*2+1,ri*2+1):
                m={}
                for arm in arms:
                    wave,rate=sf.read(listen/f"ref{ri:02}-text{ti:02}-{arm}.wav",dtype="float64")
                    m[arm]=features(wave,rate,max_hz=7600)
                hashes={indexed[ri,ti,arm]["token_sha256"] for arm in arms[:4]}
                assert len(hashes)==1
                for arm in arms:
                    row=indexed[ri,ti,arm]
                    row["spectral_to_real"]=compare(m[arm],m["real"])
                    row["spectral_to_v2proplus"]=compare(m[arm],m["v2ProPlus"])
        report["same_speaker_real_target_verified"]=True
        report["spectral_interpretation"]="Common <=7600Hz paired descriptive distances; no perceptual noise/MOS claim."
        for arm in arms:
            rows=[r for r in report["rows"] if r["arm"]==arm]
            report["summary"][arm]["mean_real_dtw_logmel_db"]=sum(r["spectral_to_real"]["dtw_logmel_l1_db"] for r in rows)/len(rows)
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
        print("Acoustic paired status",report["status"],report.get("actual_cases",0),flush=True)

if __name__=="__main__":
    main()
