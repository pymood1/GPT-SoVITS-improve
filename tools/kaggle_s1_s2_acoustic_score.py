"""Evaluate saved acoustic variants; no synthesis, no training, strict FP32."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path("/kaggle/tmp/GPT-SoVITS")
OUT=Path("/kaggle/working/acoustic-score")

def main():
    start=time.monotonic()
    report={"code_sha":REQUEST["code_sha"],"status":"failed","stage":"mount","training_performed":False,"synthesis_performed":False,"source_run":45858321277}
    OUT.mkdir(parents=True,exist_ok=False)
    try:
        def source(kernel,sha,key,count):
            paths=[p for p in Path("/kaggle/input").rglob("s1-next-result.json") if kernel in p.parts]
            assert len(paths)==1
            r=json.loads(paths[0].read_text())
            assert r["code_sha"]==sha and r["status"]=="completed" and r[key]==count
            return paths[0].parent
        kernel="gpt-sovits-s1-next-45858321277"
        assert REQUEST["source_kernel"]=="pymood1/"+kernel
        acoustic=source(kernel,"d8f0882f4a28cd70b682c2f152e5e400864cba3a","actual_cases",14)/"acoustic-cfg"
        listening=source("gpt-sovits-s1-next-45840975381","44b7d4787507da12b710c5d088d0299c2487ce32","audio_count",32)/"listen"
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
        import site
        libs=[str(p) for folder in site.getsitepackages() for p in sorted(Path(folder).glob("nvidia/*/lib")) if p.is_dir()]
        libs.extend(str(p) for p in (Path("/usr/local/cuda/lib64"),Path("/usr/lib/x86_64-linux-gnu")) if p.is_dir())
        env["LD_LIBRARY_PATH"]=os.pathsep.join(dict.fromkeys(libs+[p for p in env.get("LD_LIBRARY_PATH","").split(os.pathsep) if p]))
        report["stage"]="prepare_scores"
        texts=json.loads((listening/"texts.json").read_text())["targets"]
        arms=("saved-baseline","saved-clamp","raw-cfg050","raw-cfg080","gain-cfg080","mel-gain-only-cfg080","encoder-gain-only-cfg080","prior-teacher-cfg0","prior-teacher-ref-gain")
        request={"texts":[texts[0]],"references":[],"samples":[],"cpu_threads":2}
        for ri in (1,2):
            folder=acoustic/f"ref{ri:02}"
            ref=folder/"prior-reference-real-level.wav"
            assert ref.is_file()
            request["references"].append(str(ref))
            for arm in arms:
                path=folder/(arm+"-level.wav")
                assert path.is_file() and path.stat().st_size>44
                request["samples"].append({"label":arm,"reference_index":ri,"text_index":1,"path":str(path)})
        req=OUT/"scoring-input.json"
        req.write_text(json.dumps(request,ensure_ascii=False))
        n=len(request["samples"])
        assert n==18
        from tools.asr.fasterwhisper_asr import download_model
        download_model("large-v3")
        from tools.kaggle_s1_s2_acoustic_cfg import run_workers
        # Imported helper uses its own OUT directory; set it to this run's output.
        import tools.kaggle_s1_s2_acoustic_cfg as runner
        runner.OUT=OUT
        report["stage"]="scoring"
        asr=OUT/"asr-fp32.json";sv=OUT/"speaker-scores.json"
        cmds=[["timeout","600",sys.executable,"-u",str(ROOT/"tools/score_s1_fp32_asr.py"),"--source",str(req),"--output",str(asr),"--backend","ct2","--expected-count",str(n)],
              ["timeout","420",sys.executable,"-u",str(ROOT/"tools/score_s1_stack_waveforms.py"),str(req),"speaker",str(sv)]]
        try:
            run_workers(cmds,[dict(env,CUDA_VISIBLE_DEVICES=str(i)) for i in range(2)],["asr-fp32","speaker"],report)
        except RuntimeError:
            pass
        report["score_stages"]={}
        rows=[{"arm":r["label"],"reference_index":r["reference_index"]} for r in request["samples"]]
        for stage,path in (("asr",asr),("speaker",sv)):
            if not path.is_file():
                report["score_stages"][stage]="failed"
                continue
            d=json.loads(path.read_text())
            assert len(d["scores"])==n
            assert d["backend"]==("ct2_cuda_float32" if stage=="asr" else "cuda_fp32")
            report.setdefault("score_backends",{})[stage]=d["backend"]
            for row,r in zip(rows,d["scores"]):
                if stage=="asr":
                    assert row["arm"]==r["label"] and row["reference_index"]==r["reference_index"]
                    row.update(char_errors=r["char_errors"],char_count=r["char_count"])
                else:
                    row["speaker_cosine"]=r["speaker_cosine"]
            report["score_stages"][stage]="completed"
        report["rows"]=rows
        report["summary"]={}
        for arm in arms:
            rr=[r for r in rows if r["arm"]==arm]
            a={}
            if all("char_errors" in r for r in rr):
                a["char_errors"]=sum(r["char_errors"] for r in rr)
                a["char_count"]=sum(r["char_count"] for r in rr)
                a["cer"]=a["char_errors"]/max(1,a["char_count"])
            if all("speaker_cosine" in r for r in rr):
                a["speaker_cosine"]=sum(r["speaker_cosine"] for r in rr)/len(rr)
            report["summary"][arm]=a
        report["status"]="completed" if all(x=="completed" for x in report["score_stages"].values()) else "partial"
        report["stage"]="finished"
    except Exception as exc:
        report["error_type"]=type(exc).__name__
        import traceback
        (OUT/"private-error.log").write_text(traceback.format_exc(),encoding="utf-8")
    finally:
        report["seconds"]=round(time.monotonic()-start,3)
        Path("/kaggle/working/s1-next-result.json").write_text(json.dumps(report,allow_nan=False,indent=2))
        print("Acoustic scoring status",report["status"],flush=True)
if __name__=="__main__":
    main()
