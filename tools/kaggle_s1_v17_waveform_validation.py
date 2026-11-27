"""Exact v17 weight recovery and paired bilingual waveform validation."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path("/kaggle/tmp/GPT-SoVITS")
OUT=Path("/kaggle/working/v17-waveform-workers")

def main():
    report={"code_sha":REQUEST["code_sha"],"status":"failed","stage":"mount","training_performed":False,"cfm_synthesis_performed":True,"source_run":45952006429}
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    listen=Path("/kaggle/working/v17-waveform-validation")
    listen.mkdir()
    try:
        def source(kernel,sha,cases,key):
            paths=[p for p in Path("/kaggle/input").rglob("s1-next-result.json") if kernel in p.parts]
            assert len(paths)==1
            r=json.loads(paths[0].read_text())
            assert r["code_sha"]==sha and r["status"]=="completed" and r[key]==cases and all(v==0 for v in r["workers"].values())
            return paths[0].parent
        kernel="gpt-sovits-s1-next-45952006429"
        assert REQUEST["source_kernel"]=="pymood1/"+kernel
        prior=source(kernel,"8edca921ad16e76c9786bcc1c43ad140a06534a2",432,"actual_cases")
        en=prior/"expanded-alignment/test";ko=prior/"expanded-alignment/ko_guard"
        weights=prior/"expanded-alignment-workers"
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
        from tools.s1_v17_waveform_validation import ARMS,cases
        for lang,folder in (("en",en),("ko",ko)):
            shutil.copytree(folder,listen/lang)
        report["corpus"]={"source_run":45952006429,"en_speakers":4,"ko_speakers":2,"en_sample_rate":48000,
                         "ko_sample_rate":16000,"fixed_subset":"first/last English target per speaker; all Korean targets",
                         "reused_v17_holdout":True}
        common=[sys.executable,"-u",str(ROOT/"tools/s1_v17_waveform_validation.py"),"--en",str(en),"--ko",str(ko)]
        cmds=[common+["--weights",str(weights/f"gpu{i}"),"--output",str(OUT/f"gpu{i}"),
                "--seed",str(20261127+i)] for i in range(2)]
        report["stage"]="synthesis"
        run_workers(cmds,[dict(env,CUDA_VISIBLE_DEVICES=str(i)) for i in range(2)],["eval0","eval1"],report)
        reps=[json.loads((OUT/f"gpu{i}"/"metrics.json").read_text()) for i in range(2)]
        assert all(r["status"]=="completed" and len(r["rows"])==72 for r in reps)
        report["replications"]=reps
        report["actual_cases"]=sum(len(r["rows"]) for r in reps)
        report["expected_cases"]=144
        assert report["actual_cases"]==144
        # Cross-check regenerated tokens against v17 metrics using exact same seed/inputs.
        old=json.loads((prior/"s1-next-result.json").read_text())
        for i,r in enumerate(reps):
            lookup={(x["arm"],x["split"],x["speaker"],x["target"]):x
                    for x in old["replications"][i]["rows"]}
            for x in r["rows"]:
                if x["arm"] in ("real","oracle"):continue
                key=(x["arm"],"test" if x["language"]=="en" else "ko_guard",x["reference_index"],x["target_index"])
                prev=lookup[key]
                assert x["unit_errors"]==prev["unit_errors"] and x["length_ratio"]==prev["length_ratio"] and x["cap_hit"]==prev["cap_hit"],key
        report["regenerated_token_metric_parity"]=True
        report["stage"]="scoring"
        import site
        libs=[str(p) for folder in site.getsitepackages() for p in sorted(Path(folder).glob("nvidia/*/lib")) if p.is_dir()]
        libs.extend(str(p) for p in (Path("/usr/local/cuda/lib64"),Path("/usr/lib/x86_64-linux-gnu")) if p.is_dir())
        env["LD_LIBRARY_PATH"]=os.pathsep.join(dict.fromkeys(libs+[p for p in env.get("LD_LIBRARY_PATH","").split(os.pathsep) if p]))
        from tools.asr.fasterwhisper_asr import download_model
        download_model("large-v3")
        import statistics as st
        report["waveform_summary"]={}
        for lang,folder,count in (("en",en,96),("ko",ko,48)):
            m=json.loads((folder/"manifest.json").read_text())
            flat=cases(m,lang)
            indexes={k:j+1 for j,k in enumerate(flat)}
            req={"texts":[m["pairs"][ri-1]["targets"][ti-1]["text"] for ri,ti in flat],
                 "references":[str(folder/p["reference"]["path"]) for p in m["pairs"]],
                 "samples":[],"cpu_threads":2,"language":lang}
            rows=[]
            for i,r in enumerate(reps):
                for x in r["rows"]:
                    if x["language"]!=lang:continue
                    ri,ti,arm=x["reference_index"],x["target_index"],x["arm"]
                    stem=f"{lang}-ref{ri:02}-target{ti:02}-{arm}"
                    for kind in ("raw","level"):
                        wav=OUT/f"gpu{i}"/(stem+"-"+kind+".wav")
                        assert wav.is_file() and wav.stat().st_size>44
                        dest=listen/(f"seed{i}-"+stem+"-"+kind+".wav")
                        shutil.copyfile(wav,dest)
                    req["samples"].append(dict(label=f"seed{i}-{arm}",reference_index=ri,text_index=indexes[ri,ti],path=str(dest)))
                    rows.append(x)
            assert len(rows)==len(req["samples"])==count
            path=OUT/(lang+"-scoring.json");path.write_text(json.dumps(req,ensure_ascii=False))
            asr=OUT/(lang+"-asr.json");sv=OUT/(lang+"-sv.json")
            cmds=[["timeout","2700",sys.executable,"-u",str(ROOT/"tools/score_s1_fp32_asr.py"),"--source",str(path),"--output",str(asr),"--backend","ct2","--expected-count",str(count)],
                  ["timeout","1800",sys.executable,"-u",str(ROOT/"tools/score_s1_stack_waveforms.py"),str(path),"speaker",str(sv)]]
            run_workers(cmds,[dict(env,CUDA_VISIBLE_DEVICES=str(i)) for i in range(2)],[lang+"-asr",lang+"-speaker"],report)
            aa,bb=json.loads(asr.read_text()),json.loads(sv.read_text())
            assert aa["backend"]=="ct2_cuda_float32" and bb["backend"]=="cuda_fp32"
            assert len(aa["scores"])==len(bb["scores"])==count
            for q,x,ar,sr in zip(req["samples"],rows,aa["scores"],bb["scores"]):
                for key in ("label","reference_index","text_index"):assert q[key]==ar[key]==sr[key]
                x.update(char_errors=ar["char_errors"],char_count=ar["char_count"],speaker_cosine=sr["speaker_cosine"])
            for i in range(2):
                for arm in ARMS:
                    rr=[x for x in reps[i]["rows"] if x["language"]==lang and x["arm"]==arm]
                    assert len(rr)==count//12
                    short=sum(x["duration_ratio"]<.5 for x in rr)
                    long=sum(x["duration_ratio"]>2 for x in rr)
                    pp=[x["periodicity"]["median_periodicity"] for x in rr if x["periodicity"]["median_periodicity"] is not None]
                    hf=[x["noise"]["active_hf_ratio"] for x in rr if x["noise"]["active_hf_ratio"] is not None]
                    report["waveform_summary"][f"{lang}-seed{i}-{arm}"]=dict(cases=len(rr),
                        cer=sum(x["char_errors"] for x in rr)/sum(x["char_count"] for x in rr),
                        speaker_cosine=st.mean(x["speaker_cosine"] for x in rr),
                        median_duration_ratio=st.median(x["duration_ratio"] for x in rr),
                        median_abs_log_duration_error=st.median(abs(__import__("math").log(x["duration_ratio"])) for x in rr),
                        short=short,long=long,cap_hits=sum(x.get("cap_hit",False) for x in rr),
                        median_periodicity=st.median(pp) if pp else None,mean_active_hf=st.mean(hf) if hf else None)
        report["score_backends"]={"asr":"ct2_cuda_float32","speaker":"cuda_fp32"}
        report["caveats"]="Same v17 holdout, not fresh confirmation; real Korean source 16k, synth 48k. Oracle uses target tokens, not deployable. Token UED parity does not prove identical sequence. Periodicity/HF not MOS or noise proof."
        # Randomized filenames; retain answer key separately and exclude real/oracle.
        import random
        blind=listen/"blind-listening";blind.mkdir()
        key=[];rng=random.Random(20261127)
        for lang,flat in (("en",cases(json.loads((en/"manifest.json").read_text()),"en")),("ko",cases(json.loads((ko/"manifest.json").read_text()),"ko"))):
            for ri,ti in flat:
                options=[(i,arm) for i in range(2) for arm in ("frozen","eos2","eos4","eos4_ctc")]
                rng.shuffle(options)
                for j,(i,arm) in enumerate(options,1):
                    name=f"{lang}-ref{ri:02}-target{ti:02}-clip{j:02}.wav"
                    shutil.copyfile(listen/f"seed{i}-{lang}-ref{ri:02}-target{ti:02}-{arm}-level.wav",blind/name)
                    key.append(dict(file=name,seed=i,arm=arm))
        (listen/"blind-answer-key.json").write_text(json.dumps(key,indent=2))
        (blind/"README.txt").write_text("Compare content, omissions, reference resemblance, voiced noise/artifacts and naturalness within each ref/target group. No formal MOS. Answer key outside this folder.")
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
        print("V17 waveform validation status",report["status"],report.get("actual_cases",0),flush=True)

if __name__=="__main__":
    main()
