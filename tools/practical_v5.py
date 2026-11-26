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
        self.styles = [extract_style(self.tts, self.tts._get_ref_spec(str(self.bundle / p)))
                       for p in self.meta["auxiliary"]]
        self.lock = threading.Lock()

    def synthesize(self, text, seed=2027, reference=2, use_mean=True):
        import numpy as np
        from tools.quality_reference_style import extract_style, blend_style
        text = str(text).strip()
        if not text or len(text) > 200:
            raise ValueError("Enter 1-200 Korean characters")
        reference = int(reference)
        if reference not in (1, 2, 3):
            raise ValueError("Invalid reference")
        entry = self.meta["references"][reference - 1]
        ref = str(self.bundle / entry["file"])
        with self.lock:
            self.tts.set_ref_audio(ref)
            opts = {}
            if use_mean:
                primary = extract_style(self.tts, self.tts.prompt_cache["refer_spec"][0])
                opts["research_style_ge"] = blend_style(
                    primary, self.styles, [1 / len(self.styles)] * len(self.styles), mix=.5)
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


def export(repo, work, output):
    import subprocess
    from scipy.io.wavfile import write
    repo, work, output = Path(repo).resolve(), Path(work).resolve(), Path(output).resolve()
    setup_repo(repo)
    from tools.quality_model_mix import export_scaled_v5_checkpoint
    from tools.quality_reference_style import reference_groups
    from tools.kaggle_quality_listening import choose_references
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
            export_scaled_v5_checkpoint(work / "S2_rank32_grok_cfm_best.pth",
                                        bundle / "SoVITS-v5dev-r32-075.pth", .75)
            prompts, auxiliary, _ = reference_groups(choose_references, work)
            manifest = {
                "version": "v5dev", "s1": "upstream pretrained s1v3",
                "s2": "rank32 compact LoRA, B-factor scale 0.75",
                "style_mix": .5, "style_mode": "uniform", "default_reference": 2,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "references": [], "auxiliary": [], "sha256": {},
                "requires": ["repository runtime at git_commit", "V5dev base S2", "vocoder", "BERT", "HuBERT", "CUDA"],
            }
            for i, (path, prompt, language) in enumerate(prompts, 1):
                name = f"reference-{i}.wav"
                shutil.copyfile(path, bundle / name)
                manifest["references"].append({"file": name, "text": prompt, "language": language})
            for i, (path, _, _) in enumerate(auxiliary, 1):
                name = f"auxiliary-{i}.wav"
                shutil.copyfile(path, bundle / name)
                manifest["auxiliary"].append(name)
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
                "Korean input, up to 200 characters, 48 kHz. The default reference is 2.\n"
                "Do not publish this archive: it contains private reference recordings and transcripts.\n",
                encoding="utf-8")
            engine = PracticalV5(repo, bundle)
            for i, text in enumerate(("오늘은 새로운 목소리 모델을 확인하는 날입니다.",
                                     "정말 그렇게 생각했나요? 다시 한번 확인해 보겠습니다.",
                                     "여러 가지 방법을 비교한 후에 결정을 내렸습니다."), 1):
                sr, audio = engine.synthesize(text, seed=8200 + i, reference=i)
                write(bundle / f"smoke-{i}.wav", sr, audio)
                status["smoke_count"] += 1
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
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    if args.export_work:
        export(args.repo, args.export_work, args.output)
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
                gr.Checkbox(label="Mean 스타일 결합", value=True)],
        outputs=gr.Audio(label="합성 음성", type="numpy"),
        title="V5dev 실사용 모델")
    demo.queue().launch(share=args.share)


if __name__ == "__main__":
    main()
