"""Multi-hypothesis S1/S2 screening v23."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path("/kaggle/tmp/GPT-SoVITS")
OUT=Path("/kaggle/working/multi-v23-workers")

def main():
    report={"code_sha":REQUEST["code_sha"],"status":"failed","stage":"mount","training_performed":True,"cfm_synthesis_performed":True,"source_run":45952006429}
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    listen=Path("/kaggle/working/multi-v23")
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
        from tools.s1_s2_multi_v23 import ARMS_EN,ARMS_KO
        old21=source("gpt-sovits-s1-next-45990209360","5dcabc1fecb748457022f93c1a9a80da6dcbaf92",144,"actual_cases")
        ar_weights=old21/"word-alignment-v21-workers"
        corpus=old21/"word-alignment-v21"
        for name in ("en_val","en_extra"):
            shutil.copytree(corpus/name,listen/name)
        en=listen/"en";val=listen/"en_val";extra=listen/"en_extra";ko=listen/"ko"
        old22=source("gpt-sovits-s1-next-46001023562","e3ba6695d2cbbe2ad31e5279646a6af8dfb88a46",144,"actual_cases")
        nar_weights=old22/"masked-v22-workers"
        from tools.prepare_s1_expanded_vctk import prepare as prepare_en
        from tools.prepare_s1_korean_replay import prepare as prepare_ko
        def exclusions(folders):
            pairs=[p for folder in folders for p in json.loads((folder/"manifest.json").read_text())["pairs"]]
            items=[r for p in pairs for r in (p["reference"],*p["targets"])]
            return dict(excluded_speakers=[p["speaker_id"] for p in pairs],excluded_texts=[r["text"] for r in items],excluded_hashes=[r["sha256"] for r in items])
        prepare_en(en,4,4,revision="0492fa00cf72afb16e753e95d0ec3439eebb07a6",max_shards=64,
            **exclusions([train,prior/"expanded-alignment/validation",prior/"expanded-alignment/test",corpus/"en",val,extra]))
        prepare_ko(ko,4,2,revision="1fe937899f828af822293d05e086200946088bdf",split="train",
            **exclusions([kt,prior/"expanded-alignment/ko_guard",corpus/"ko"]))
        assert len(json.loads((ko/"manifest.json").read_text())["pairs"])==4
        from huggingface_hub import snapshot_download
        name="google-bert/bert-base-multilingual-cased"
        bert_revision="3f076fdb1ab68d5b2880cb87a0886f315b8146f8"
        bert=snapshot_download(name,revision=bert_revision,allow_patterns=["config.json","model.safetensors","tokenizer.json","tokenizer_config.json","vocab.txt","special_tokens_map.json","pytorch_model.bin"])
        report["bert"]={"name":name,"revision":bert_revision,"alignment":"word_g2p_exact_with_nearest_fallback","encoder_frozen":True}
        report["corpus"]={"en_fresh_relative_to_v17_v20":True,"en_training_original_utterances":160,"en_train_speakers":32,
            "en_test_speakers":4,"ko_training_utterances":32,"ko_test_speakers":4,
            "ko_fresh_relative_to_v17_v19":True,"ko_source_split":"train","ko_sample_rate":16000,"en_sample_rate":48000,"pretraining_overlap":"unknown"}
        common=[sys.executable,"-u",str(ROOT/"tools/s1_s2_multi_v23.py"),"--en-train",str(train),"--en-extra",str(extra),
                "--en-val",str(val),"--en-test",str(en),"--ko-train",str(kt),"--ko-test",str(ko),
                "--bert-model",str(bert),"--steps","400"]
        cmds=[common+["--weights",str(weights/f"gpu{i}"),"--ar-weights",str(ar_weights/f"gpu{i}"),"--nar-weights",str(nar_weights/f"gpu{i}"),"--output",str(OUT/f"gpu{i}"),
                "--seed",str(20261127+i)] for i in range(2)]
        report["stage"]="dual_track"
        run_workers(cmds,[dict(env,CUDA_VISIBLE_DEVICES=str(i)) for i in range(2)],["eval0","eval1"],report)
        reps=[json.loads((OUT/f"gpu{i}"/"metrics.json").read_text()) for i in range(2)]
        assert all(r["status"]=="completed" and len(r["rows"])==184 and len(r["diagnostic"])==32 for r in reps)
        report["replications"]=reps
        report["actual_cases"]=sum(len(r["rows"]) for r in reps)
        report["expected_cases"]=368
        assert report["actual_cases"]==368
        for r in reps:
            for ri in range(1,5):
                for ti in (1,2):
                    rr=[x for x in r["rows"] if x["language"]=="ko" and x["reference_index"]==ri and x["target_index"]==ti]
                    assert len(rr)==10 and len({x["token_sha256"] for x in rr})==1
        report["ko_fixed_token_parity"]=True
        def cases(m,lang):
            return [(ri,ti) for ri,p in enumerate(m["pairs"],1) for ti in ((1,4) if lang=="en" else (1,2))]
        report["stage"]="scoring"
        import site
        libs=[str(p) for folder in site.getsitepackages() for p in sorted(Path(folder).glob("nvidia/*/lib")) if p.is_dir()]
        libs.extend(str(p) for p in (Path("/usr/local/cuda/lib64"),Path("/usr/lib/x86_64-linux-gnu")) if p.is_dir())
        env["LD_LIBRARY_PATH"]=os.pathsep.join(dict.fromkeys(libs+[p for p in env.get("LD_LIBRARY_PATH","").split(os.pathsep) if p]))
        from tools.asr.fasterwhisper_asr import download_model
        download_model("large-v3")
        import statistics as st
        report["waveform_summary"]={}
        for lang,folder,count in (("en",en,208),("ko",ko,160)):
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
                for key in ("label","reference_index","text_index"):assert q[key]==ar[key]
                # Existing speaker scorer emits positional scores only, in request order.
                assert "speaker_cosine" in sr
                x.update(char_errors=ar["char_errors"],char_count=ar["char_count"],speaker_cosine=sr["speaker_cosine"])
            for i in range(2):
                for arm in (ARMS_EN if lang=="en" else ARMS_KO):
                    rr=[x for x in reps[i]["rows"] if x["language"]==lang and x["arm"]==arm]
                    assert len(rr)==8
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
        report["caveats"]="EN and KO new speakers/text/audio relative to v17/v20/v21 listed splits; KO drawn from public train split but speaker-disjoint from adaptation; pretraining overlap unknown. AR/NAR comparison not matched compute or pretraining. KO fixed target token parity; gain and trim vary reference condition. Periodicity/HF not perceptual noise proof."
        # Randomized filenames; retain answer key separately and exclude real/oracle.
        import random
        blind=listen/"blind-listening";blind.mkdir()
        key=[];rng=random.Random(20261127)
        for lang,flat in (("en",cases(json.loads((en/"manifest.json").read_text()),"en")),("ko",cases(json.loads((ko/"manifest.json").read_text()),"ko"))):
            for ri,ti in flat:
                options=[(i,arm) for i in range(2) for arm in (ARMS_EN if lang=="en" else ARMS_KO)]
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
        print("Multi-hypothesis v23 status",report["status"],report.get("actual_cases",0),flush=True)

if __name__=="__main__":
    main()
