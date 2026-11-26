"""Private listening comparison using identical reference and text for all models."""
import json
import os
import traceback
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


class ReferenceUnavailable(RuntimeError):
    pass


def choose_references(folder, diagnostics=None, count=1, exclude=()):
    """Prefer Korean; permit an accurately transcribed other-language prompt."""
    import wave
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.update(rows=0, readable=0, duration_eligible=0, text_eligible=0, korean_eligible=0)
    candidates = []
    for row in (folder / "2-name2text.txt").read_text(encoding="utf-8").splitlines():
        diagnostics["rows"] += 1
        parts = row.split("\t", 3)
        if len(parts) != 4:
            continue
        name, _, _, transcript = parts
        if Path(name).name != name:
            continue
        path = folder / "5-wav32k" / name
        try:
            with wave.open(str(path), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
        except (OSError, EOFError, wave.Error):
            continue
        diagnostics["readable"] += 1
        # The canonical TTS accepts 3-10 seconds. Keep a small boundary margin.
        if not 3.05 <= duration <= 9.95:
            continue
        diagnostics["duration_eligible"] += 1
        transcript = transcript.strip()
        if len(transcript) < 2 or not re.search(r"\w", transcript):
            continue
        diagnostics["text_eligible"] += 1
        korean = bool(re.search("[가-힣]", transcript))
        diagnostics["korean_eligible"] += int(korean)
        # auto uses the canonical language segmenter for other or mixed scripts.
        language = "all_ko" if korean and not re.search("[A-Za-zぁ-ゖァ-ヺ一-龯]", transcript) else "auto"
        candidates.append((not korean, abs(duration - 5.0), name, path, transcript, language))
    if not candidates:
        raise ReferenceUnavailable("No transcribed reference within the supported duration")
    candidates.sort()
    unique = {}
    for candidate in candidates:
        unique.setdefault(candidate[2], candidate)
    candidates = [c for c in unique.values() if c[3] not in exclude]
    if not candidates:
        raise ReferenceUnavailable("No unused eligible reference files")
    # Distinct prompt files, deterministic selection across short/mid/long lengths.
    chosen = [candidates.pop(0)]
    for duration in (3.8, 8.0):
        if len(chosen) >= count or not candidates:
            break
        candidate = min(candidates, key=lambda x: (x[0], abs(_wave_duration(x[3]) - duration), x[2]))
        candidates.remove(candidate)
        chosen.append(candidate)
    if len(chosen) != count:
        raise ReferenceUnavailable("Insufficient distinct reference files")
    return [(path, transcript, language) for _, _, _, path, transcript, language in chosen]


def _wave_duration(path):
    import wave
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def choose_reference(folder, diagnostics=None):
    return choose_references(folder, diagnostics, 1)[0]


def normalized_text(text):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def character_errors(expected, actual):
    """Return edit count and reference character count, excluding whitespace/punctuation."""
    expected, actual = normalized_text(expected), normalized_text(actual)
    row = list(range(len(actual) + 1))
    for i, a in enumerate(expected, 1):
        nxt = [i]
        for j, b in enumerate(actual, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = nxt
    return row[-1], len(expected)


def progress(phase, label, done, total):
    worker = os.environ.get("GPT_SOVITS_MIX_WORKER", "")
    worker = "gpu" + worker if worker in ("0", "1") else "parent"
    print(f"[gsv-progress] {worker} {phase} {label} {done}/{total}", flush=True)


def run_score_stage(output, stage, request, *, cpu_only=False):
    """GPU runtime failures stay inside a subprocess; retry the entire stage on CPU."""
    import subprocess
    import time
    started = time.monotonic()
    source = output / "scoring-input.json"
    target = output / (stage + "-scores.json")
    for device in (("cpu",) if cpu_only else ("cuda", "cpu")):
        target.unlink(missing_ok=True)
        log = output / (stage + "-" + device + "-error.log")
        try:
            # stdout holds only counters, while raw diagnostics remain private.
            with log.open("w", encoding="utf-8") as error:
                p = subprocess.run([sys.executable, "-u", str(ROOT / "tools/kaggle_quality_score.py"),
                                    str(source), stage, device, str(target)],
                                   cwd=ROOT, stderr=error, timeout=1800)
            if p.returncode != 0 or not target.is_file():
                continue
            result = json.loads(target.read_text(encoding="utf-8"))
            if len(result["scores"]) != len(request["samples"]):
                continue
            result["total_seconds"] = round(time.monotonic() - started, 3)
            return result
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
            continue
    return None


def apply_score_stage(models, evaluation, stage, request, result):
    if result is None:
        evaluation[stage + "_status"] = "failed"
        return
    for item, score in zip(request["samples"], result["scores"]):
        sample = next(s for s in models[item["label"]]["samples"] if s["index"] == item["index"])
        sample.update(score)
    evaluation.update({stage + "_status": "completed",
                       stage + "_backend": result["backend"],
                       stage + "_seconds": result["total_seconds"],
                       stage + "_calibration_seconds": result["calibration_seconds"],
                       stage + "_calibration_count": result["calibration_count"],
                       stage + "_cpu_calibration_seconds": result.get("cpu_calibration_seconds", 0),
                       stage + "_gpu_calibration_seconds": result.get("gpu_calibration_seconds", 0)})


def evaluate_quality(output, reference, models, texts=TEXTS, references=None, heldout=None):
    """Score completed synthesis on idle T4; CPU parity and fallback preserve comparability."""
    import os
    request = {"samples": [
        {"label": label, "index": s["index"], "text_index": s.get("text_index", s["index"]),
         "reference_index": s.get("reference_index", 1),
         "path": str(output / f"{label}_{s['index']:02d}.wav")}
        for label, result in models.items() for s in result["samples"]],
        "texts": list(texts), "references": [str(p) for p in (references or [reference])],
        "heldout_references": [str(p) for p in (heldout or [])],
        "cpu_threads": 2 if os.environ.get("GPT_SOVITS_MIX_WORKER") else 4}
    evaluation = {"asr_status": "unavailable", "speaker_status": "unavailable"}
    if not request["samples"]:
        return evaluation
    (output / "scoring-input.json").write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    for stage in ("asr", "speaker"):
        result = run_score_stage(output, stage, request)
        apply_score_stage(models, evaluation, stage, request, result)
    return evaluation


def reconcile_score_backends(reports, folders):
    """If one lane fell back, re-score the other on CPU before comparing candidates."""
    for stage in ("asr", "speaker"):
        if len(reports) != 2 or any(r.get("evaluation", {}).get(stage + "_status") != "completed" for r in reports):
            continue
        backends = [r["evaluation"].get(stage + "_backend") for r in reports]
        if backends[0] == backends[1]:
            continue
        for index, backend in enumerate(backends):
            if not isinstance(backend, str) or not backend.startswith("cuda_"):
                continue
            folder = folders[index]
            report = reports[index]
            request = json.loads((folder / "scoring-input.json").read_text(encoding="utf-8"))
            old_seconds = report["evaluation"].get(stage + "_seconds", 0)
            result = run_score_stage(folder, stage, request, cpu_only=True)
            apply_score_stage(report["models"], report["evaluation"], stage, request, result)
            if result is None:
                report["status"] = "failed"
            else:
                report["evaluation"][stage + "_seconds"] += old_seconds
            (folder / "listening-metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


def pick_candidate(arms, names, metric):
    import math
    valid = []
    for name in names:
        arm = arms.get(name, {})
        value = (arm.get("best") or {}).get(metric)
        if arm.get("returncode") == 0 and type(value) in (int, float) and math.isfinite(value):
            valid.append((value, name))
    if not valid:
        raise RuntimeError("No valid trained candidate")
    return min(valid)[1]


MIX_LABELS = (("mix025", "mix050", "mix075", "mix125"),
              ("base", "rank32", "cfg100", "cfg160"))
SOFT_LABELS = (("soft000", "soft005", "soft010", "soft025"),
               ("soft050", "soft100", "mix065", "mix085", "soft000b"))


GENERAL_LABELS = (("gen_base", "gen_hard", "gen_soft010"),
                  ("gen_hardb", "gen_soft010b", "gen_soft025"))
STYLE_LABELS = (("style_base", "style_base_mean", "style_base_sv"),
                ("style_lora", "style_lora_mean", "style_lora_sv"))


def merge_mix_reports(reports, labels=MIX_LABELS, sample_count=15):
    """Require complete, disjoint matched evaluations before accepting a sweep."""
    result = {"status": "failed", "models": {}, "evaluation": {}}
    if len(reports) != 2:
        return result
    complete = True
    for index, report in enumerate(reports):
        models = report.get("models", {})
        expected = set(labels[index])
        complete = complete and report.get("status") == "completed" and set(models) == expected
        result["models"].update({k: models[k] for k in expected if k in models})
        for label in expected:
            model = models.get(label, {})
            samples = model.get("samples", [])
            complete = complete and model.get("status") == "completed" and len(samples) == sample_count
            complete = complete and {s.get("index") for s in samples} == set(range(1, sample_count + 1))
            complete = complete and all("speaker_cosine" in s and "cer" in s for s in samples)
            if labels == STYLE_LABELS:
                complete = complete and all(isinstance(s.get("speaker_holdout_cosine"), (int, float)) for s in samples)
    for key in ("asr_status", "speaker_status"):
        result["evaluation"][key] = "completed" if all(r.get("evaluation", {}).get(key) == "completed" for r in reports) else "failed"
        complete = complete and result["evaluation"][key] == "completed"
        stage = key.removesuffix("_status")
        backends = [r.get("evaluation", {}).get(stage + "_backend") for r in reports]
        if any(backends):
            complete = complete and len(set(backends)) == 1
            result["evaluation"][stage + "_backend"] = backends[0] if len(set(backends)) == 1 else "mixed"
            result["evaluation"][stage + "_seconds"] = max(r.get("evaluation", {}).get(stage + "_seconds", 0) for r in reports)
            result["evaluation"][stage + "_calibration_count"] = sum(r.get("evaluation", {}).get(stage + "_calibration_count", 0) for r in reports)
            for suffix in ("_cpu_calibration_seconds", "_gpu_calibration_seconds"):
                result["evaluation"][stage + suffix] = sum(r.get("evaluation", {}).get(stage + suffix, 0) for r in reports)

    if reports[0].get("reference_counts") != reports[1].get("reference_counts"):
        complete = False
    result["reference_counts"] = reports[0].get("reference_counts", {})
    if complete:
        result["status"] = "completed"
    return result


def run_mix_workers(work, output):
    """Both workers share read-only preprocessed inputs, never writable configs."""
    import subprocess
    import time
    output.mkdir(parents=True, exist_ok=True)
    workers = []
    reports = []
    deadline = time.monotonic() + 9000
    try:
        for index in range(2):
            folder = output / f"gpu{index}"
            folder.mkdir(exist_ok=True)
            env = os.environ.copy()
            env.update(GPT_SOVITS_MIX_WORKER=str(index), CUDA_VISIBLE_DEVICES=str(index),
                       _CUDA_VISIBLE_DEVICES=str(index), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            workers.append((subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                              str(work), str(folder)], cwd=ROOT, env=env), folder))
        while any(p.poll() is None for p, _ in workers) and time.monotonic() < deadline:
            time.sleep(10)
    finally:
        for p, _ in workers:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        for p, folder in workers:
            try:
                report = json.loads((folder / "listening-metrics.json").read_text())
            except (OSError, ValueError):
                report = {"status": "failed"}
            if p.returncode != 0:
                report["status"] = "failed"
            reports.append(report)
        profile = os.environ.get("GPT_SOVITS_RESEARCH_PROFILE")
        labels = STYLE_LABELS if profile == "quality_style" else GENERAL_LABELS if profile == "quality_general" else SOFT_LABELS if profile == "quality_soft" else MIX_LABELS
        reconcile_score_backends(reports, [folder for _, folder in workers])
        result = merge_mix_reports(reports, labels, 18 if profile in ("quality_general", "quality_style") else 15)
        (output / "listening-metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "completed" else 1


def main():
    if os.environ.get("GPT_SOVITS_RESEARCH_PROFILE") in ("quality_mix", "quality_soft", "quality_general", "quality_style") and not os.environ.get("GPT_SOVITS_MIX_WORKER"):
        return run_mix_workers(*map(Path, sys.argv[1:3]))
    import numpy as np
    import torch
    from scipy.io.wavfile import write
    from TTS_infer_pack.TTS import TTS, TTS_Config

    work, output = map(Path, sys.argv[1:3])
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "models": {}}
    try:
        base_s1 = ROOT / "GPT_SoVITS/pretrained_models/s1v3.ckpt"
        base_s2 = ROOT / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"
        models = {"base": (base_s1, base_s2)}
        profile = os.environ.get("GPT_SOVITS_RESEARCH_PROFILE")
        reuse = profile in ("quality_reuse", "quality_mix", "quality_soft", "quality_general", "quality_style")
        mix_worker = os.environ.get("GPT_SOVITS_MIX_WORKER", "")
        inference_options = {}
        if profile == "quality_style":
            from tools.quality_model_mix import export_scaled_v5_checkpoint
            if mix_worker not in ("0", "1"):
                raise ValueError("Invalid style worker")
            weight = base_s2
            if mix_worker == "1":
                weight = work / "style-anchor.pth"
                export_scaled_v5_checkpoint(work / "S2_rank32_grok_cfm_best.pth", weight, .75)
            models = {label: (base_s1, weight) for label in STYLE_LABELS[int(mix_worker)]}
            inference_options = {label: {"soft_semantic_mix": 0.} for label in models}
            reference_folder = work
            torch.set_num_threads(2)
        elif profile == "quality_general":
            from tools.quality_model_mix import export_scaled_v5_checkpoint
            if mix_worker not in ("0", "1"):
                raise ValueError("Invalid generalization worker")
            weight = work / ("general-anchor-" + mix_worker + ".pth")
            export_scaled_v5_checkpoint(work / "S2_rank32_grok_cfm_best.pth", weight, .75)
            models = {}
            for label in GENERAL_LABELS[int(mix_worker)]:
                models[label] = (base_s1, base_s2 if label == "gen_base" else weight)
                inference_options[label] = {"soft_semantic_mix": .25 if label == "gen_soft025" else .10 if "soft010" in label else 0.}
            reference_folder = work
            torch.set_num_threads(2)
        elif profile == "quality_soft":
            from tools.quality_model_mix import export_scaled_v5_checkpoint
            s32 = work / "S2_rank32_grok_cfm_best.pth"
            if mix_worker not in ("0", "1"):
                raise ValueError("Invalid soft worker")
            # Each process has its own compact checkpoint to avoid write races.
            weight = work / ("soft-anchor-" + mix_worker + ".pth")
            export_scaled_v5_checkpoint(s32, weight, .75)
            models = {}
            if mix_worker == "0":
                for label, mix in (("soft000", 0.), ("soft005", .05),
                                   ("soft010", .10), ("soft025", .25)):
                    models[label] = (base_s1, weight)
                    inference_options[label] = {"soft_semantic_mix": mix}
            else:
                for label, mix in (("soft050", .50), ("soft100", 1.), ("soft000b", 0.)):
                    models[label] = (base_s1, weight)
                    inference_options[label] = {"soft_semantic_mix": mix}
                for label, scale in (("mix065", .65), ("mix085", .85)):
                    candidate = work / (label + ".pth")
                    export_scaled_v5_checkpoint(s32, candidate, scale)
                    models[label] = (base_s1, candidate)
            reference_folder = work
            torch.set_num_threads(2)
        elif profile == "quality_mix":
            from tools.quality_model_mix import export_scaled_v5_checkpoint
            s32 = work / "S2_rank32_grok_cfm_best.pth"
            models = {}
            if mix_worker == "0":
                for label, scale in (("mix025", .25), ("mix050", .50), ("mix075", .75), ("mix125", 1.25)):
                    weight = work / (label + ".pth")
                    export_scaled_v5_checkpoint(s32, weight, scale)
                    models[label] = (base_s1, weight)
            elif mix_worker == "1":
                models.update(base=(base_s1, base_s2), rank32=(base_s1, s32),
                              cfg100=(base_s1, s32), cfg160=(base_s1, s32))
                inference_options.update(cfg100={"cfg_rate": 1.0}, cfg160={"cfg_rate": 1.6})
            else:
                raise ValueError("Invalid mix worker")
            reference_folder = work
            torch.set_num_threads(2)
        elif reuse:
            s1 = work / "S1_qkv_o-grok-best.ckpt"
            s16 = work / "S2_rank16_grok_cfm_best.pth"
            s32 = work / "S2_rank32_grok_cfm_best.pth"
            models.update(rank16=(base_s1, s16), rank32=(base_s1, s32), combined=(s1, s32))
            reference_folder = work
        elif os.environ.get("GPT_SOVITS_RESEARCH_PROFILE") == "quality_session":
            with __import__("zipfile").ZipFile("/kaggle/working/grokking_result.zip") as z:
                summary = json.loads(z.read("training/summary.json"))
            best_s1 = pick_candidate(summary["arms"], ("S1_v_o", "S1_qkv_o"), "val_loss")
            best_s2 = pick_candidate(summary["arms"], ("S2_rank16", "S2_rank32"), "val_cfm")
            s1_weight = work / best_s1 / "weights" / (best_s1 + "-grok-best.ckpt")
            s2_weight = work / best_s2 / "weights" / (best_s2 + "_grok_cfm_best.pth")
            if not s1_weight.is_file() or not s2_weight.is_file():
                raise FileNotFoundError("Selected compact checkpoint absent")
            report["selected"] = {"s1": best_s1, "s2": best_s2}
            models.update(s1_only=(s1_weight, base_s2), s2_only=(base_s1, s2_weight),
                          combined=(s1_weight, s2_weight))
            reference_folder = work / best_s2
        else:
            for arm in ("A_lr100", "B_lr030"):
                path = work / arm / "weights" / (arm + "_grok_cfm_best.pth")
                if not path.is_file():
                    raise FileNotFoundError("Best checkpoint absent")
                destination = output / (arm + ".pth")
                shutil.copyfile(path, destination)
                models[arm] = (base_s1, destination)
            reference_folder = work / "A_lr100"
        report["reference_counts"] = {}
        references = choose_references(reference_folder, report["reference_counts"], 3 if reuse else 1)
        texts = TEXTS + ("아니, 잠깐만요. 그건 제가 생각했던 것과 완전히 다른데요?", "지금부터 하나씩 차근차근 확인해 보겠습니다.") if reuse else TEXTS
        seeds = (1234,)
        auxiliary = heldout = []
        sv_weights = None
        if profile == "quality_style":
            from tools.quality_reference_style import reference_groups, speaker_weights, extract_style, blend_style
            references, auxiliary, heldout = reference_groups(choose_references, reference_folder, replication=True)
            sv_weights = speaker_weights(references, auxiliary)
        if profile in ("quality_general", "quality_style"):
            if profile == "quality_general":
                previous = {r[0] for r in references}
                references = choose_references(reference_folder, report["reference_counts"], 3, exclude=previous)
            texts = ("오늘은 평소보다 조금 일찍 출발하려고 합니다.", "정말 그렇게 생각했어요? 저는 전혀 예상하지 못했는데요.", "여러 가지 방법을 비교해 보고 가장 적절한 것을 선택했습니다.")
            seeds = (2027, 3141)
            if profile == "quality_style":
                texts = ("오늘 오후에는 잠시 시간을 내서 산책을 하려고 합니다.", "잠깐만요, 정말 그런 일이 있었다는 말인가요?", "지금까지 확인한 내용을 바탕으로 다음 계획을 세워 보겠습니다.")
                seeds = (4099, 7919)
        cases = [(i, text, seed) for seed in seeds for i, text in enumerate(texts, 1)]
        # Keep exact prompts privately; no transcript enters the public report.
        for ref_index, (ref, prompt, prompt_lang) in enumerate(references, 1):
            stem = f"reference-{ref_index:02d}" if reuse else "reference"
            shutil.copyfile(ref, output / (stem + ".wav"))
            (output / (stem + ".json")).write_text(json.dumps({"text": prompt, "language": prompt_lang}, ensure_ascii=False), encoding="utf-8")
        ref = references[0][0]
        for label, (gpt_weight, weight) in models.items():
            result = {"samples": [], "status": "failed"}
            report["models"][label] = result
            try:
                result["stage"] = "model_load"
                progress("model_load", label, 0, len(references) * len(cases))
                config = dict(TTS_Config.default_configs["v5dev"], device="cuda:0",
                              is_half=False, t2s_weights_path=str(gpt_weight), vits_weights_path=str(weight))
                if not Path(gpt_weight).is_file() or not Path(weight).is_file():
                    raise FileNotFoundError("Explicit candidate checkpoint missing")
                tts_config = TTS_Config({"custom": config})
                tts_config.configs_path = str(output / "tts-private.yaml")
                tts = TTS(tts_config)
                style_mode = "sv" if label.endswith("_sv") else "uniform" if label.endswith("_mean") else None
                aux_styles = [extract_style(tts, tts._get_ref_spec(str(r[0]))) for r in auxiliary] if style_mode else []
                for ref_index, (ref, prompt, prompt_lang) in enumerate(references, 1):
                    result["stage"] = "reference"
                    tts.set_ref_audio(str(ref))
                    style_override = {}
                    if style_mode:
                        primary = extract_style(tts, tts.prompt_cache["refer_spec"][0])
                        weights = sv_weights[ref_index - 1] if style_mode == "sv" else [1 / len(aux_styles)] * len(aux_styles)
                        style_override = {"research_style_ge": blend_style(primary, aux_styles, weights)}
                    for case_index, (text_index, text, seed) in enumerate(cases, 1):
                        index = (ref_index - 1) * len(cases) + case_index
                        result["stage"] = "synthesis"
                        progress("synthesis", label, len(result["samples"]), len(references) * len(cases))
                        chunks = list(tts.run({
                            "text": text, "text_lang": "all_ko", "prompt_text": prompt,
                            "prompt_lang": prompt_lang, "ref_audio_path": str(ref),
                            "sample_steps": 32, "cfg_rate": 1.30, "seed": seed,
                            "batch_size": 1, "parallel_infer": False,
                            "use_cuda_graph": False, "use_flash_attention": False,
                            "text_split_method": "cut0", "split_bucket": False,
                            "return_fragment": False, "streaming_mode": False,
                            "cache_dit_mode": "off",
                            **inference_options.get(label, {}),
                            **style_override,
                        }))
                        if len(chunks) != 1:
                            raise RuntimeError("Unexpected synthesis chunks")
                        sr, audio = chunks[0]
                        audio = np.asarray(audio)
                        result["stage"] = "output_validation"
                        result["observed_sample_rate"] = int(sr)
                        expected_sr = int(tts.vocoder_configs["sr"])
                        if sr != expected_sr or audio.size < sr or not np.isfinite(audio).all():
                            raise RuntimeError("Invalid synthesis output")
                        normalized = audio.astype(np.float64)
                        if np.issubdtype(audio.dtype, np.integer):
                            normalized /= max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
                        rms = float(np.sqrt(np.mean(normalized ** 2)))
                        if rms < 1e-4:
                            raise RuntimeError("Silent synthesis output")
                        write(output / f"{label}_{index:02d}.wav", sr, audio)
                        result["samples"].append({"index": index, "reference_index": ref_index, "text_index": text_index, "seed": seed, "sample_rate": int(sr), "seconds": audio.size / sr,
                                                  "rms": rms, "peak": float(np.max(np.abs(normalized)))})
                progress("synthesis", label, len(result["samples"]), len(references) * len(cases))
                result.update(status="completed", stage="finished")
                print(f"[quality] {label}: completed {len(result['samples'])} samples", flush=True)
            except Exception as exc:
                progress("failed", label, len(result["samples"]), len(references) * len(cases))
                result["error_type"] = type(exc).__name__
                # Raw traceback is private Kaggle output, never returned to GitHub.
                (output / (label + "-error.txt")).write_text(traceback.format_exc(), encoding="utf-8")
            finally:
                if "tts" in locals():
                    del tts
                torch.cuda.empty_cache()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        report["evaluation"] = evaluate_quality(output, references[0][0], report["models"], texts, [r[0] for r in references], [r[0] for r in heldout])
        complete = all(report["models"].get(k, {}).get("status") == "completed"
                       and len(report["models"][k]["samples"]) == len(references) * len(cases)
                       for k in models)
        if reuse:
            complete = complete and all(report["evaluation"].get(k) == "completed"
                                        for k in ("asr_status", "speaker_status"))
            complete = complete and all("cer" in sample and "speaker_cosine" in sample
                                        for model in report["models"].values() for sample in model["samples"])
        if complete:
            report["status"] = "completed"
        progress("finished" if complete else "failed", "all", int(complete), 1)
    except Exception as exc:
        report["error_type"] = type(exc).__name__
    finally:
        (output / "listening-metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
