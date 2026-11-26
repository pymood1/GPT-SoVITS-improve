"""Private V5dev deployment export and single-user inference launcher."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import zipfile
from pathlib import Path


def setup_repo(repo):
    repo = Path(repo).resolve()
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo / "GPT_SoVITS")]
    os.environ["version"] = "v5dev"
    os.environ.setdefault("language", "ko_KR")
    return repo


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mounted_sources(root, candidate_kernel, reference_kernel):
    """Use only the two explicitly pinned private kernel outputs."""
    import re
    import wave
    def slug(value):
        if not re.fullmatch(r"pymood1/gpt-sovits-research-[0-9]+", value):
            raise ValueError("Invalid source kernel")
        return value.split("/")[-1]
    root = Path(root)
    candidate_slug, reference_slug = slug(candidate_kernel), slug(reference_kernel)
    candidates = [p for p in root.rglob("S2_rank32_grok_cfm_best.pth")
                  if candidate_slug in p.parts and p.parent.name == "private-candidates"
                  and p.is_file()]
    refs = [p.parent for p in root.rglob("reference-01.json")
            if reference_slug in p.parts and p.parent.name == "gpu0"
            and p.parent.parent.name == "private-listening"]
    if len(candidates) != 1 or len(refs) != 1:
        raise FileNotFoundError("Expected unique pinned checkpoint and reference folder")
    checkpoint = candidates[0]
    with checkpoint.open("rb") as f:
        if f.read(2) != b"07":
            raise ValueError("Incorrect V5dev checkpoint header")
    prompts, hashes = [], set()
    for i in range(1, 4):
        wav = refs[0] / f"reference-{i:02d}.wav"
        meta = json.loads((refs[0] / f"reference-{i:02d}.json").read_text(encoding="utf-8"))
        if not isinstance(meta.get("text"), str) or not meta["text"].strip() or meta.get("language") not in ("auto", "all_ko"):
            raise ValueError("Invalid saved reference transcript")
        with wave.open(str(wav), "rb") as f:
            duration = f.getnframes() / f.getframerate()
        digest = sha256(wav)
        if not 3 <= duration <= 10 or digest in hashes:
            raise ValueError("Invalid or duplicate saved reference")
        hashes.add(digest)
        prompts.append((wav, meta["text"], meta["language"]))
    return checkpoint, prompts


class PracticalV5:
    def __init__(self, repo, bundle):
        setup_repo(repo)
        from TTS_infer_pack.TTS import TTS, TTS_Config
        from tools.quality_reference_style import extract_style
        self.bundle = Path(bundle).resolve()
        self.meta = json.loads((self.bundle / "manifest.json").read_text(encoding="utf-8"))
        for name, digest in self.meta["sha256"].items():
            if sha256(self.bundle / name) != digest:
                raise ValueError("Bundle integrity check failed")
        self.config_dir = tempfile.TemporaryDirectory(prefix="practical-v5-config-")
        config = dict(TTS_Config.default_configs["v5dev"], device="cuda:0", is_half=False,
                      t2s_weights_path=str(self.bundle / "GPT-s1v3.ckpt"),
                      vits_weights_path=str(self.bundle / "SoVITS-v5dev-r32-075.pth"))
        cfg = TTS_Config({"custom": config})
        cfg.configs_path = str(Path(self.config_dir.name) / "tts.yaml")
        self.tts = TTS(cfg)
        self.styles = [extract_style(self.tts, self.tts._get_ref_spec(str(self.bundle / p["file"])))
                       for p in self.meta["references"]]
        self.lock = threading.Lock()

    def synthesize(self, text, seed=2027, reference=2, use_mean=False):
        import numpy as np
        from tools.quality_reference_style import extract_style, blend_style
        text = str(text).strip()
        if not text or len(text) > 200:
            raise ValueError("Enter 1-200 Korean characters")
        reference = int(reference)
        if use_mean and not self.meta.get("mean_available", True):
            raise ValueError("Mean did not pass deployment validation")
        if reference not in (1, 2, 3):
            raise ValueError("Invalid reference")
        entry = self.meta["references"][reference - 1]
        ref = str(self.bundle / entry["file"])
        with self.lock:
            self.tts.set_ref_audio(ref)
            opts = {}
            if use_mean:
                primary = extract_style(self.tts, self.tts.prompt_cache["refer_spec"][0])
                other = [style for i, style in enumerate(self.styles, 1) if i != reference]
                opts["research_style_ge"] = blend_style(
                    primary, other, [1 / len(other)] * len(other), mix=.5)
            chunks = list(self.tts.run({
                "text": text, "text_lang": "all_ko", "prompt_text": entry["text"],
                "prompt_lang": entry["language"], "ref_audio_path": ref,
                "sample_steps": 32, "cfg_rate": 1.30, "seed": int(seed),
                "batch_size": 1, "parallel_infer": False, "use_cuda_graph": False,
                "use_flash_attention": False, "text_split_method": "cut0",
                "split_bucket": False, "return_fragment": False, "streaming_mode": False,
                "cache_dit_mode": "off", "soft_semantic_mix": 0., **opts,
            }))
            if len(chunks) != 1:
                raise RuntimeError("Expected one synthesis output")
            sr, audio = chunks[0]
            audio = np.asarray(audio)
            if sr != 48000 or audio.size < sr or not np.isfinite(audio).all():
                raise RuntimeError("Invalid audio output")
            norm = audio.astype(np.float64)
            if np.issubdtype(audio.dtype, np.integer):
                info = np.iinfo(audio.dtype)
                norm /= max(abs(info.min), info.max)
            if np.sqrt(np.mean(norm ** 2)) < 1e-4:
                raise RuntimeError("Silent output")
            return int(sr), audio


def export(repo, work, output, *, candidate_kernel, reference_kernel, input_root='/kaggle/input'):
    import subprocess
    from scipy.io.wavfile import write
    repo, work, output = Path(repo).resolve(), Path(work).resolve(), Path(output).resolve()
    setup_repo(repo)
    from tools.quality_model_mix import export_scaled_v5_checkpoint
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "deploy-status.json"
    status = {"status": "failed", "smoke_count": 0}
    target = output / "V5dev-practical-private.zip"
    target.unlink(missing_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="private-v5-export-", dir="/kaggle/tmp") as tmp:
            bundle = Path(tmp) / "V5dev-practical"
            bundle.mkdir()
            shutil.copyfile(repo / "GPT_SoVITS/pretrained_models/s1v3.ckpt", bundle / "GPT-s1v3.ckpt")
            checkpoint, prompts = mounted_sources(input_root, candidate_kernel, reference_kernel)
            export_scaled_v5_checkpoint(checkpoint, bundle / "SoVITS-v5dev-r32-075.pth", .75)
            manifest = {
                "version": "v5dev", "s1": "upstream pretrained s1v3",
                "s2": "rank32 compact LoRA, B-factor scale 0.75",
                "style_mix": .5, "style_mode": "uniform_other_two_optional", "default_reference": 2,
                "default_mean": False, "candidate_kernel": candidate_kernel, "reference_kernel": reference_kernel,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "references": [], "sha256": {},
                "requires": ["repository runtime at git_commit", "V5dev base S2", "vocoder", "BERT", "HuBERT", "CUDA"],
            }
            for i, (path, prompt, language) in enumerate(prompts, 1):
                name = f"reference-{i}.wav"
                shutil.copyfile(path, bundle / name)
                manifest["references"].append({"file": name, "text": prompt, "language": language})
            shutil.copyfile(repo / "tools/practical_v5.py", bundle / "practical_v5.py")
            for file in sorted(bundle.iterdir()):
                manifest["sha256"][file.name] = sha256(file)
            (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            (bundle / "README.txt").write_text(
                "V5dev practical candidate (private).\n"
                "Install the repository runtime and its pretrained assets first; compact LoRA is not standalone.\n"
                "Use the exact git_commit in manifest.json. Run from an initialized CUDA environment:\n"
                "python practical_v5.py --repo /path/to/GPT-SoVITS --bundle /path/to/V5dev-practical\n"
                "Local WebUI by default; --share explicitly enables a public Gradio tunnel.\n"
                "Korean input, up to 200 characters, 48 kHz. Default reference 2; Mean off.\n"
                "Optional Mean pools the other two saved references; it is not the original three-auxiliary experiment.\n"
                "Do not publish this archive: it contains private reference recordings and transcripts.\n",
                encoding="utf-8")
            # No training ASR ran in this path, so explicitly provision the scorer.
            from tools.asr.fasterwhisper_asr import download_model
            download_model("large-v3")
            estimate = sum(p.stat().st_size for p in bundle.iterdir() if p.is_file())
            if shutil.disk_usage(output).free < estimate + 100 * 1024**2:
                raise RuntimeError("Insufficient free output space for private archive")
            engine = PracticalV5(repo, bundle)
            texts = ("오늘은 새로운 목소리 모델을 확인하는 날입니다.",
                     "정말 그렇게 생각했나요? 다시 한번 확인해 보겠습니다.",
                     "여러 가지 방법을 비교한 후에 결정을 내렸습니다.")
            samples = []
            for mean in (False, True):
                label = "mean" if mean else "baseline"
                for i, text in enumerate(texts, 1):
                    sr, audio = engine.synthesize(text, seed=8200 + i, reference=i, use_mean=mean)
                    path = bundle / f"smoke-{label}-{i}.wav"
                    write(path, sr, audio)
                    samples.append({"label": label, "index": i, "text_index": i,
                                    "reference_index": i, "path": str(path)})
                    status["smoke_count"] += 1
            del engine
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            from tools.kaggle_quality_listening import run_score_stage
            request = {"samples": samples, "texts": list(texts), "cpu_threads": 2}
            score_dir = Path(tmp) / "score"
            score_dir.mkdir()
            (score_dir / "scoring-input.json").write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
            result = run_score_stage(score_dir, "asr", request)
            if result is None:
                raise RuntimeError("ASR smoke verification failed")
            for label in ("baseline", "mean"):
                scores = [v for sample, v in zip(samples, result["scores"]) if sample["label"] == label]
                cer = sum(v["char_errors"] for v in scores) / max(1, sum(v["char_count"] for v in scores))
                status[label + "_cer"] = cer
            if status["baseline_cer"] > .05:
                raise RuntimeError("Baseline CER exceeds deployment threshold")
            manifest["mean_smoke_cer"] = status["mean_cer"]
            manifest["baseline_smoke_cer"] = status["baseline_cer"]
            manifest["mean_available"] = status["mean_cer"] <= .05
            (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            # Zip only after loading and synthesizing from the exported weights.
            partial = target.with_suffix(".partial")
            try:
                with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as z:
                    for file in sorted(bundle.iterdir()):
                        z.write(file, "V5dev-practical/" + file.name)
                with zipfile.ZipFile(partial) as z:
                    if z.testzip() is not None:
                        raise ValueError("Archive integrity check failed")
                    count = len(z.namelist())
                os.replace(partial, target)
            finally:
                partial.unlink(missing_ok=True)
            status.update(status="completed", file_count=count, archive_bytes=target.stat().st_size)
    except Exception as exc:
        status["error_type"] = type(exc).__name__
        import traceback
        (output / "private-deploy-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--bundle")
    parser.add_argument("--export-work")
    parser.add_argument("--output")
    parser.add_argument("--candidate-kernel")
    parser.add_argument("--reference-kernel")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    if args.export_work:
        export(args.repo, args.export_work, args.output, candidate_kernel=args.candidate_kernel,
               reference_kernel=args.reference_kernel)
        return
    if not args.bundle:
        parser.error("--bundle is required")
    bundle = Path(args.bundle).resolve()
    engine = PracticalV5(args.repo, bundle)
    import gradio as gr
    demo = gr.Interface(fn=engine.synthesize,
        inputs=[gr.Textbox(label="한국어 합성 문장", lines=4),
                gr.Number(label="시드", value=2027, precision=0),
                gr.Radio([1, 2, 3], label="기준 음성", value=2),
                gr.Checkbox(label="Mean 스타일 결합 (선택)", value=False, interactive=engine.meta.get("mean_available", False))],
        outputs=gr.Audio(label="합성 음성", type="numpy"),
        title="V5dev 실사용 모델")
    demo.queue().launch(share=args.share)


if __name__ == "__main__":
    main()
