"""Korean fixed-token factorial v24."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path("/kaggle/tmp/GPT-SoVITS")
OUT=Path("/kaggle/working/ko-factorial-v24-workers")

def main():
    report={"code_sha":REQUEST["code_sha"],"status":"failed","stage":"mount","training_performed":False,"cfm_synthesis_performed":True,"source_run":45952006429}
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    listen=Path("/kaggle/working/ko-factorial-v24")
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
        train=prior/"expanded-alignment/train";kt=prior/"expanded-alignment/ko_train"
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
        from tools.s1_s2_ko_factorial_v24 import ARMS
        old21=source("gpt-sovits-s1-next-45990209360","5dcabc1fecb748457022f93c1a9a80da6dcbaf92",144,"actual_cases")
        old23=source("gpt-sovits-s1-next-46006831134","fd87872f858a8e26a2da85244ef1f1cdbb4df783",368,"actual_cases")
        from tools.prepare_s1_korean_replay import prepare as prepare_ko
        ko=listen/"ko"
        folders=[kt,prior/"expanded-alignment/ko_guard",old21/"word-alignment-v21/ko",old23/"multi-v23/ko"]
        pairs=[p for folder in folders for p in json.loads((folder/"manifest.json").read_text())["pairs"]]
        items=[r for p in pairs for r in (p["reference"],*p["targets"])]
        prepare_ko(ko,8,2,revision="1fe937899f828af822293d05e086200946088bdf",split="train",
            excluded_speakers=[p["speaker_id"] for p in pairs],excluded_texts=[r["text"] for r in items],excluded_hashes=[r["sha256"] for r in items])
        assert len(json.loads((ko/"manifest.json").read_text())["pairs"])==8
        report["corpus"]={"ko_test_speakers":8,"ko_fresh_relative_to_v17_v19_v23":True,"ko_source_split":"train","ko_sample_rate":16000,"pretraining_overlap":"unknown"}
        report["arms"]=ARMS
        common=[sys.executable,"-u",str(ROOT/"tools/s1_s2_ko_factorial_v24.py"),"--ko-test",str(ko)]
        cmds=[common+["--weights",str(weights/f"gpu{i}"),"--output",str(OUT/f"gpu{i}"),"--seed",str(20261129+i)] for i in range(2)]
        report["stage"]="dual_track"
        run_workers(cmds,[dict(env,CUDA_VISIBLE_DEVICES=str(i)) for i in range(2)],["eval0","eval1"],report)
        reps=[json.loads((OUT/f"gpu{i}"/"metrics.json").read_text()) for i in range(2)]
        assert all(r["status"]=="completed" and len(r["rows"])==160 for r in reps)
        report["replications"]=reps
        report["actual_cases"]=sum(len(r["rows"]) for r in reps)
        report["expected_cases"]=320
        assert report["actual_cases"]==320
        for r in reps:
            for ri in range(1,9):
                for ti in (1,2):
                    rr=[x for x in r["rows"] if x["language"]=="ko" and x["reference_index"]==ri and x["target_index"]==ti]
                    assert len(rr)==10 and len({x["token_sha256"] for x in rr})==1
        report["ko_fixed_token_parity"]=True
        def cases(m,lang):
            return [(ri,ti) for ri,p in enumerate(m["pairs"],1) for ti in (1,2)]
        report["stage"]="scoring"
        import site
        libs=[str(p) for folder in site.getsitepackages() for p in sorted(Path(folder).glob("nvidia/*/lib")) if p.is_dir()]
        libs.extend(str(p) for p in (Path("/usr/local/cuda/lib64"),Path("/usr/lib/x86_64-linux-gnu")) if p.is_dir())
        env["LD_LIBRARY_PATH"]=os.pathsep.join(dict.fromkeys(libs+[p for p in env.get("LD_LIBRARY_PATH","").split(os.pathsep) if p]))
        from tools.asr.fasterwhisper_asr import download_model
        download_model("large-v3")
        import statistics as st
        report["waveform_summary"]={}
        for lang,folder,count in (("ko",ko,320),):
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
            cmds=[["timeout","4500",sys.executable,"-u",str(ROOT/"tools/score_s1_fp32_asr.py"),"--source",str(path),"--output",str(asr),"--backend","ct2","--expected-count",str(count)],
                  ["timeout","2700",sys.executable,"-u",str(ROOT/"tools/score_s1_stack_waveforms.py"),str(path),"speaker",str(sv)]]
            run_workers(cmds,[dict(env,CUDA_VISIBLE_DEVICES=str(i)) for i in range(2)],[lang+"-asr",lang+"-speaker"],report)
            aa,bb=json.loads(asr.read_text()),json.loads(sv.read_text())
            assert aa["backend"]=="ct2_cuda_float32" and bb["backend"]=="cuda_fp32"
            assert len(aa["scores"])==len(bb["scores"])==count
            for q,x,ar,sr in zip(req["samples"],rows,aa["scores"],bb["scores"]):
                for key in ("label","reference_index","text_index"):assert q[key]==ar[key]
                # Existing speaker scorer emits positional scores only, in request order.
                assert "speaker_cosine" in sr
                x.update(char_errors=ar["char_errors"],char_count=ar["char_count"],speaker_cosine=sr["speaker_cosine"])
            for i in range(2):
                for arm in ARMS:
                    rr=[x for x in reps[i]["rows"] if x["language"]==lang and x["arm"]==arm]
                    assert len(rr)==16
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
                        median_periodicity=st.median(pp) if pp else None,mean_active_hf=st.mean(hf) if hf else None, mean_generation_seconds=st.mean(x["generation_seconds"] for x in rr), mean_conditional_evals=st.mean(x["conditional_evals"] for x in rr), mean_negative_evals=st.mean(x["negative_evals"] for x in rr))
        report["score_backends"]={"asr":"ct2_cuda_float32","speaker":"cuda_fp32"}
        report["caveats"]="Fresh relative to listed Zeroth speaker/text/audio splits, historical pretraining overlap unknown. Fixed target and prompt semantic tokens; gain alters S2 reference acoustics only. Same per-case seed, noise scale deliberately differs in one arm. HF and periodicity are not listening scores. Eight speaker clusters; no deployment selection."
        # Randomized filenames; retain answer key separately and exclude real/oracle.
        import random
        blind=listen/"blind-listening";blind.mkdir()
        key=[];rng=random.Random(20261127)
        for lang,flat in (("ko",cases(json.loads((ko/"manifest.json").read_text()),"ko")),):
            for ri,ti in flat:
                options=[(i,arm) for i in range(2) for arm in ARMS]
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
        print("Korean factorial v24 status",report["status"],report.get("actual_cases",0),flush=True)
        if report["status"]=="completed":
            from tools.kaggle_s1_live_status import emit_summary
            emit_summary(report)

if __name__=="__main__":
    main()
