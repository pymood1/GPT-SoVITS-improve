"""Private listening comparison using identical reference and text for all models."""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "GPT_SoVITS")]
TEXTS = (
    "오늘은 날씨가 좋아서 잠깐 산책을 하고 왔어요.",
    "정말 그렇게 생각해? 나는 조금 다른 의견이 있어.",
    "처음에는 어려웠지만, 천천히 연습하면 충분히 할 수 있어요.",
)


def choose_reference(folder):
    import soundfile as sf
    candidates = []
    for row in (folder / "2-name2text.txt").read_text(encoding="utf-8").splitlines():
        parts = row.split("\t", 3)
        if len(parts) != 4:
            continue
        name, _, _, transcript = parts
        if Path(name).name != name or not re.search("[가-힣]", transcript):
            continue
        path = folder / "5-wav32k" / name
        try:
            info = sf.info(path)
        except (OSError, RuntimeError):
            continue
        duration = info.duration
        if 3.2 <= duration <= 8.0 and len(transcript.strip()) >= 5:
            candidates.append((abs(duration - 5.0), name, path, transcript.strip()))
    if not candidates:
        raise RuntimeError("No suitable Korean reference")
    _, _, path, transcript = min(candidates)
    return path, transcript


def main():
    import numpy as np
    import torch
    from scipy.io.wavfile import write
    from TTS_infer_pack.TTS import TTS, TTS_Config

    work, output = map(Path, sys.argv[1:3])
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "models": {}}
    try:
        ref, prompt = choose_reference(work / "A_lr100")
        models = {"base": ROOT / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"}
        for arm in ("A_lr100", "B_lr030"):
            path = work / arm / "weights" / (arm + "_grok_cfm_best.pth")
            if not path.is_file():
                raise FileNotFoundError("Best checkpoint absent")
            destination = output / (arm + ".pth")
            shutil.copyfile(path, destination)
            models[arm] = destination
        for label, weight in models.items():
            result = {"samples": [], "status": "failed"}
            report["models"][label] = result
            try:
                config = dict(TTS_Config.default_configs["v5dev"], device="cuda:0",
                              is_half=False, vits_weights_path=str(weight))
                tts = TTS(TTS_Config({"custom": config}))
                for index, text in enumerate(TEXTS, 1):
                    chunks = list(tts.run({
                        "text": text, "text_lang": "all_ko", "prompt_text": prompt,
                        "prompt_lang": "all_ko", "ref_audio_path": str(ref),
                        "sample_steps": 32, "cfg_rate": 1.30, "seed": 1234,
                        "batch_size": 1, "parallel_infer": False,
                        "use_cuda_graph": False, "use_flash_attention": False,
                        "text_split_method": "cut0", "split_bucket": False,
                        "return_fragment": False, "streaming_mode": False,
                        "cache_dit_mode": "off",
                    }))
                    if len(chunks) != 1:
                        raise RuntimeError("Unexpected synthesis chunks")
                    sr, audio = chunks[0]
                    audio = np.asarray(audio)
                    if sr != 32000 or audio.size < sr or not np.isfinite(audio).all():
                        raise RuntimeError("Invalid synthesis output")
                    normalized = audio.astype(np.float64)
                    if np.issubdtype(audio.dtype, np.integer):
                        normalized /= max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
                    rms = float(np.sqrt(np.mean(normalized ** 2)))
                    if rms < 1e-4:
                        raise RuntimeError("Silent synthesis output")
                    write(output / f"{label}_{index:02d}.wav", sr, audio)
                    result["samples"].append({"index": index, "seconds": audio.size / sr,
                                              "rms": rms, "peak": float(np.max(np.abs(normalized)))})
                result["status"] = "completed"
            except Exception as exc:
                result["error_type"] = type(exc).__name__
            finally:
                if "tts" in locals():
                    del tts
                torch.cuda.empty_cache()
        if all(report["models"].get(k, {}).get("status") == "completed" for k in models):
            report["status"] = "completed"
    except Exception as exc:
        report["error_type"] = type(exc).__name__
    finally:
        (output / "listening-metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
