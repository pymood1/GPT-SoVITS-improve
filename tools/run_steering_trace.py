"""Controlled steering trace runner. --self-test uses stdlib-only CPU mocks."""
import argparse
from array import array
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
import wave


ROOT = Path(__file__).resolve().parents[1]
SENTENCES = (
    "상자는 문 옆에 있다.",
    "표의 두 번째 줄에는 무엇이 적혀 있나요?",
    "이 종이는 책상 위에 놓아 주세요.",
    "복도를 따라가면 창문이 세 개 있고, 끝에는 작은 회의실이 있습니다.",
    "오전 열 시에 자료를 나누고, 항목별로 번호를 붙인 다음, 날짜와 수량을 차례대로 확인하는 것으로 정리하겠습니다.",
)
EMOTIONS = ("happy", "angry", "disgusted", "surprised")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="CPU mocks; no torch/TTS import")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only; no model/file loading or output writes")
    parser.add_argument("--limit", type=int, help="Run only the first N requests (positive; omit for full plan)")
    parser.add_argument("--controller")
    parser.add_argument("--s1-checkpoint")
    parser.add_argument("--s2-checkpoint")
    parser.add_argument("--reference-audio")
    parser.add_argument("--reference-text")
    parser.add_argument("--reference-language", help="An existing TTS prompt_lang code, e.g. all_ko")
    parser.add_argument("--config", default=str(ROOT / "GPT_SoVITS/configs/tts_infer.yaml"))
    parser.add_argument("--sentences-json", help="Optional JSON list of exactly five distinct sentences")
    parser.add_argument("--emotions", nargs="+", default=list(EMOTIONS))
    parser.add_argument("--strengths", nargs="+", type=float, default=[1.0, 2.0])
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.35)
    parser.add_argument("--seed", type=int, default=233333)
    parser.add_argument("--device", help="Override WebUI-style default: cuda when available, otherwise cpu")
    parser.add_argument("--output-root", default="/kaggle/working/steering_audio")
    return parser


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan(sentences, emotions, strengths, emotion_names, limit=None):
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    if (
        not isinstance(sentences, (tuple, list)) or len(sentences) != 5
        or not all(isinstance(text, str) and text.strip() for text in sentences)
        or len(set(text.strip() for text in sentences)) != 5
    ):
        raise ValueError("Exactly five distinct nonempty sentences are required")
    if not emotion_names or len(set(emotion_names)) != len(emotion_names):
        raise ValueError("Checkpoint emotion_names must be present and unique")
    if not emotions or len(set(emotions)) != len(emotions):
        raise ValueError("Requested emotions must be present and unique")
    if set(emotions) - set(emotion_names):
        raise ValueError("Requested emotion is absent from checkpoint metadata")
    if (
        not strengths or len(set(strengths)) != len(strengths)
        or any(not math.isfinite(value) or value <= 0 for value in strengths)
    ):
        raise ValueError("Trace strengths must be unique, finite and positive")
    plan = []
    for sentence_id, sentence in enumerate(sentences, 1):
        for emotion in emotions:
            for strength in strengths:
                plan.append({
                    "index": len(plan) + 1,
                    "sentence_id": sentence_id,
                    "sentence": sentence,
                    "emotion": emotion,
                    "emotion_id": emotion_names.index(emotion),
                    "strength": float(strength),
                })
    return plan if limit is None else plan[:limit]


def validate_options(args):
    if not 0 <= args.seed < 2**32:
        raise ValueError("A fixed seed in 0..2**32-1 is required; -1 is random")
    if args.top_k < 2 or not 0 < args.top_p <= 1:
        raise ValueError("Use top-k >= 2 and top-p in (0, 1]")
    for value in (args.temperature, args.repetition_penalty):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Temperature/repetition penalty must be finite and positive")
    for name in ("controller", "s1_checkpoint", "s2_checkpoint", "reference_audio", "config"):
        value = getattr(args, name)
        if not value or not Path(value).is_file():
            raise ValueError(f"Missing input file: --{name.replace('_', '-')}")
        setattr(args, name, str(Path(value).resolve()))
    if not args.reference_text or not args.reference_language:
        raise ValueError("Reference text and language must be supplied")
    sentences = SENTENCES
    if args.sentences_json:
        args.sentences_json = str(Path(args.sentences_json).resolve())
        sentences = json.loads(Path(args.sentences_json).read_text(encoding="utf-8"))
    return sentences


def webui_runtime_defaults(cuda_available, device_override=None):
    """Match WebUI's ordinary True/False is_half environment setting safely."""
    value = os.environ.get("is_half", "True").lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError("is_half must be True/False or 1/0; Python expressions are unsupported")
    device = device_override or ("cuda" if cuda_available else "cpu")
    return {"device": device, "is_half": value in {"true", "1"} and cuda_available
            and str(device).split(":")[0] != "cpu"}


def check_trace_runtime(runtime, actual, fixed):
    expected = {
        "probability_dtype": actual["dtype"], "score_dtype": actual["dtype"],
        "device_type": actual["device"].split(":")[0],
        "sampling": {key: fixed[key] for key in
                     ("top_k", "top_p", "temperature", "repetition_penalty")},
    }
    if runtime != expected:
        raise ValueError(f"Trace runtime differs from loaded model/settings: {runtime!r} != {expected!r}")


def create_pipeline(args, run_dir):
    """Real backend; never called by --self-test. Existing TTS owns pairing checks."""
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "GPT_SoVITS"))
    if "_CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
    from AR.models.emotion_control import STEERING_TRACE, load_emotion_controller_checkpoint
    if not STEERING_TRACE:
        raise ValueError("Trace module was imported OFF; start this runner in a fresh process")
    _, metadata = load_emotion_controller_checkpoint(args.controller, map_location="cpu")
    if metadata.get("extra", {}).get("s1_sha256") != file_hash(args.s1_checkpoint):
        raise ValueError("Missing or mismatched Controller/S1 SHA-256 binding")
    version = metadata["extra"].get("version")
    if version not in {"v5dev", "v5turbo"}:
        raise ValueError("Checkpoint metadata must specify v5dev or v5turbo")
    sentences = SENTENCES if not args.sentences_json else json.loads(
        Path(args.sentences_json).read_text(encoding="utf-8")
    )
    build_plan(sentences, args.emotions, args.strengths, metadata.get("emotion_names"), args.limit)
    import yaml
    import torch
    from TTS_infer_pack.TTS import TTS, TTS_Config, set_seed
    config_data = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(config_data, dict):
        raise ValueError("TTS config must contain a mapping")
    custom = dict(config_data.get("custom") or config_data.get(version) or {})
    custom.update({"version": version, "t2s_weights_path": args.s1_checkpoint,
                   "vits_weights_path": args.s2_checkpoint})
    custom.update(webui_runtime_defaults(torch.cuda.is_available(), args.device))
    config_data["custom"] = custom
    config = TTS_Config(config_data)
    if args.reference_language not in config.languages:
        raise ValueError("Reference language is not supported by this TTS config")
    # init_t2s/init_vits save this config. Never overwrite the user's source YAML.
    config.configs_path = str(run_dir / "tts_runtime.yaml")
    pipeline = TTS(config)
    metadata = pipeline.load_emotion_controller(args.controller)
    return pipeline, metadata, set_seed


class TraceTee:
    """Forward stderr unchanged; count observed trace events without rewriting them."""
    def __init__(self, original, handle):
        self.original, self.handle = original, handle
        self.pending = ""
        self.begin_request()

    def begin_request(self):
        self.trace_steps = 0
        self.generations = set()
        self.runtimes = []

    def __getattr__(self, name):
        return getattr(self.original, name)

    def write(self, text):
        # tqdm can leave an unterminated progress line on stderr. JSON must
        # start on its own line for the existing line-oriented aggregator.
        if self.pending and text.startswith('{"event":'):
            self.original.write("\n")
            self.handle.write("\n")
            self.pending = ""
        self.original.write(text)
        self.handle.write(text)
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            for part in line.split("\r"):
                try:
                    event = json.loads(part)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event.get("event") == "emotion_steering":
                    self.trace_steps += 1
                    self.generations.add(str(event.get("generation")))
                    runtime = event.get("runtime")
                    if runtime not in self.runtimes:
                        self.runtimes.append(runtime)
        return len(text)

    def flush(self):
        self.original.flush()
        self.handle.flush()


def write_wave(path, sample_rate, audio):
    view = memoryview(audio)
    if view.ndim != 1 or view.itemsize != 2 or view.format.lstrip("@=<>!") != "h":
        raise ValueError("Expected mono signed PCM16 output from TTS.run")
    if not len(view) or sample_rate <= 0:
        raise ValueError("TTS returned empty audio or invalid sample rate")
    data = view.tobytes()
    if sys.byteorder != "little":
        samples = array("h")
        samples.frombytes(data)
        samples.byteswap()
        data = samples.tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(data)


def save_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_experiment(args, pipeline_factory=create_pipeline):
    sentences = validate_options(args)
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=root))
    manifest_path = run_dir / "steering_manifest.json"
    manifest = {
        "status": "initializing", "requests": [],
        "backend": "existing_TTS.run" if pipeline_factory is create_pipeline else "injected_for_test",
        "quality_evaluation": "not_performed",
    }
    current = None
    with (run_dir / "steering_trace.log").open("w", encoding="utf-8") as handle:
        tee = TraceTee(sys.stderr, handle)
        with contextlib.redirect_stderr(tee):
            try:
                os.environ["GPT_SOVITS_STEERING_TRACE"] = "1"
                save_manifest(manifest_path, manifest)
                pipeline, metadata, seed_setter = pipeline_factory(args, run_dir)
                if metadata.get("extra", {}).get("s1_sha256") != file_hash(args.s1_checkpoint):
                    raise ValueError("Missing or mismatched Controller/S1 SHA-256 binding")
                plan = build_plan(sentences, args.emotions, args.strengths, metadata.get("emotion_names"), args.limit)
                fixed = {
                    "text_lang": "all_ko", "ref_audio_path": args.reference_audio,
                    "prompt_text": args.reference_text, "prompt_lang": args.reference_language,
                    "top_k": args.top_k, "top_p": args.top_p,
                    "temperature": args.temperature, "repetition_penalty": args.repetition_penalty,
                    "seed": args.seed, "text_split_method": "cut1", "batch_size": 1,
                    "return_fragment": False, "streaming_mode": False,
                    "parallel_infer": False, "split_bucket": False,
                }
                weight = pipeline.t2s_model.model.ar_predict_layer.weight
                actual_runtime = {
                    "dtype": str(weight.dtype), "device": str(weight.device),
                    "is_half": bool(pipeline.configs.is_half),
                    "batch_size": fixed["batch_size"], "text_split_method": fixed["text_split_method"],
                }
                manifest.update({
                    "status": "running", "emotion_order": metadata["emotion_names"],
                    "controller_sha256": file_hash(args.controller),
                    "s1_sha256": file_hash(args.s1_checkpoint),
                    "s2_file_sha256": file_hash(args.s2_checkpoint),
                    "fixed_inputs": {key: value for key, value in fixed.items() if key != "prompt_text"},
                    "reference_text_sha256": hashlib.sha256(args.reference_text.encode()).hexdigest(),
                    "requests": [{**row, "status": "pending"} for row in plan],
                    "actual_runtime": actual_runtime,
                    "limit": args.limit,
                })
                save_manifest(manifest_path, manifest)
                for current in manifest["requests"]:
                    current["status"] = "running"
                    save_manifest(manifest_path, manifest)
                    tee.begin_request()
                    print(json.dumps({"event": "request_marker", **{
                        key: current[key] for key in ("index", "sentence_id", "emotion", "strength")
                    }}), file=sys.stderr, flush=True)
                    seed_setter(args.seed)
                    outputs = list(pipeline.run({
                        **fixed, "text": current["sentence"],
                        "emotion_id": current["emotion_id"], "emotion_strength": current["strength"],
                    }))
                    if len(outputs) != 1:
                        raise ValueError("Expected one non-streaming TTS output per request")
                    if tee.trace_steps == 0:
                        raise ValueError("No steering trace observed; request cannot be a measured success")
                    for runtime in tee.runtimes:
                        check_trace_runtime(runtime, actual_runtime, fixed)
                    manifest["trace_runtime"] = tee.runtimes[0]
                    sample_rate, audio = outputs[0]
                    filename = (
                        f"{current['index']:03d}_s{current['sentence_id']}"
                        f"_e{current['emotion_id']}_strength{current['strength']:g}.wav"
                    )
                    write_wave(run_dir / filename, sample_rate, audio)
                    current.update({"status": "complete", "wav": filename,
                                    "sample_rate": int(sample_rate), "trace_steps": tee.trace_steps,
                                    "trace_generations": sorted(tee.generations)})
                    save_manifest(manifest_path, manifest)
                manifest["status"] = "complete"
            except BaseException as error:
                manifest.update({"status": "failed", "error_type": type(error).__name__})
                if current is not None and current["status"] == "running":
                    current["status"] = "failed"
                raise
            finally:
                save_manifest(manifest_path, manifest)
    return run_dir


def run_self_tests():
    class RunnerTests(unittest.TestCase):
        def make_args(self, root):
            args = make_parser().parse_args([])
            for name in ("controller", "s1_checkpoint", "s2_checkpoint", "reference_audio", "config"):
                path = root / name
                path.write_bytes(b"synthetic fixture")
                setattr(args, name, str(path))
            args.reference_text = "참조 문장입니다."
            args.reference_language = "all_ko"
            args.output_root = str(root / "outputs")
            return args

        def factory(self, args, run_dir, fail_at=None, with_trace=True):
            from types import SimpleNamespace as NS
            self.assertEqual(os.environ.get("GPT_SOVITS_STEERING_TRACE"), "1")
            calls, seeds, random_values = [], [], []

            class Pipeline:
                configs = NS(is_half=False)
                t2s_model = NS(model=NS(ar_predict_layer=NS(weight=NS(dtype="torch.float32", device="cpu"))))
                def run(_, inputs):
                    calls.append(dict(inputs))
                    random_values.append(random.random())
                    if fail_at == len(calls):
                        raise ValueError("mock failure")
                    if with_trace:
                        print(json.dumps({"event": "emotion_steering", "generation": len(calls),
                                          "runtime": {"probability_dtype": "torch.float32",
                                                      "score_dtype": "torch.float32", "device_type": "cpu",
                                                      "sampling": {k: inputs[k] for k in
                                                                   ("top_k", "top_p", "temperature", "repetition_penalty")}}}),
                              file=sys.stderr, flush=True)
                    yield 24000, array("h", [0, 12, -12, 0])

            def seed_setter(value):
                seeds.append(value)
                random.seed(value)

            self.calls, self.seeds, self.random_values = calls, seeds, random_values
            metadata = {"emotion_names": list(reversed(EMOTIONS)),
                        "extra": {"s1_sha256": file_hash(args.s1_checkpoint), "version": "v5dev"}}
            return Pipeline(), metadata, seed_setter

        def test_plan_order_and_metadata_ids(self):
            plan = build_plan(SENTENCES, EMOTIONS, (1.0, 2.0), list(reversed(EMOTIONS)))
            self.assertEqual(len(plan), 40)
            self.assertEqual(plan[0], {"index": 1, "sentence_id": 1, "sentence": SENTENCES[0],
                                       "emotion": "happy", "emotion_id": 3, "strength": 1.0})
            self.assertEqual((plan[1]["strength"], plan[8]["sentence_id"]), (2.0, 2))
            self.assertEqual((plan[-1]["index"], plan[-1]["sentence_id"], plan[-1]["emotion"]),
                             (40, 5, "surprised"))

        def test_complete_mock_run_markers_manifest_seed_and_wavs(self):
            with tempfile.TemporaryDirectory() as folder:
                args = self.make_args(Path(folder))
                with contextlib.redirect_stderr(io.StringIO()) as stderr:
                    directory = run_experiment(args, self.factory)
                manifest = json.loads((directory / "steering_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["actual_runtime"], {
                    "dtype": "torch.float32", "device": "cpu", "is_half": False,
                    "batch_size": 1, "text_split_method": "cut1"})
                self.assertEqual(manifest["trace_runtime"]["probability_dtype"], "torch.float32")
                self.assertEqual(len(self.calls), 40)
                self.assertEqual(self.seeds, [args.seed] * 40)
                self.assertEqual(len(set(self.random_values)), 1)
                events = [json.loads(line) for line in stderr.getvalue().splitlines()]
                self.assertEqual([event["event"] for event in events],
                                 ["request_marker", "emotion_steering"] * 40)
                self.assertEqual([event["index"] for event in events if event["event"] == "request_marker"],
                                 list(range(1, 41)))
                self.assertEqual((directory / "steering_trace.log").read_text(encoding="utf-8"), stderr.getvalue())
                for request, inputs in zip(manifest["requests"], self.calls):
                    self.assertEqual(request["status"], "complete")
                    self.assertEqual(inputs["seed"], args.seed)
                    for key, value in {"batch_size": 1, "text_split_method": "cut1",
                                       "parallel_infer": False, "split_bucket": False}.items():
                        self.assertEqual(inputs[key], value)
                        self.assertEqual(manifest["fixed_inputs"][key], value)
                    self.assertEqual(inputs["ref_audio_path"], str(Path(args.reference_audio).resolve()))
                    self.assertEqual(inputs["prompt_text"], args.reference_text)
                    for key, value in {"top_k": 15, "top_p": 1.0, "temperature": 1.0,
                                       "repetition_penalty": 1.35}.items():
                        self.assertEqual(inputs[key], value)
                    with wave.open(str(directory / request["wav"]), "rb") as handle:
                        self.assertEqual((handle.getframerate(), handle.getnframes()), (24000, 4))

        def test_failure_preserves_partial_manifest(self):
            with tempfile.TemporaryDirectory() as folder:
                args = self.make_args(Path(folder))
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, "mock failure"):
                        run_experiment(args, lambda a, d: self.factory(a, d, fail_at=2))
                path = next(Path(args.output_root).glob("*/steering_manifest.json"))
                manifest = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "failed")
                self.assertEqual([row["status"] for row in manifest["requests"][:3]],
                                 ["complete", "failed", "pending"])

        def test_limit_smoke_run_and_dry_run(self):
            with tempfile.TemporaryDirectory() as folder:
                args = self.make_args(Path(folder))
                args.limit = 2
                with contextlib.redirect_stderr(io.StringIO()):
                    directory = run_experiment(args, self.factory)
                manifest = json.loads((directory / "steering_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(len(self.calls), 2)
                self.assertEqual(manifest["limit"], 2)
                self.assertEqual([row["index"] for row in manifest["requests"]], [1, 2])
                self.assertEqual(manifest["status"], "complete")
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(main(["--dry-run", "--limit", "2"]), 0)
                self.assertEqual(len(json.loads(output.getvalue())["requests"]), 2)
                for limit in (0, -1):
                    with self.assertRaisesRegex(ValueError, "--limit must be positive"):
                        build_plan(SENTENCES, EMOTIONS, (1., 2.), EMOTIONS, limit)

        def test_dry_run_without_model_inputs_or_side_effects(self):
            from unittest.mock import patch
            previous_env = dict(os.environ)
            with patch("__main__.run_experiment", side_effect=AssertionError("must not load")):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(main(["--dry-run"]), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(len(report["requests"]), 40)
            self.assertEqual(report["requests"][0]["sentence"], SENTENCES[0])
            self.assertEqual(report["requests"][-1]["index"], 40)
            self.assertEqual({row["seed"] for row in report["requests"]}, {233333})
            self.assertTrue(all("emotion_id" not in row for row in report["requests"]))
            self.assertEqual(dict(os.environ), previous_env)

        def test_progress_line_cannot_prefix_json_record(self):
            original, log = io.StringIO(), io.StringIO()
            tee = TraceTee(original, log)
            tee.write("\rprogress 1/100")
            tee.write('{"event": "emotion_steering", "generation": 1}')
            tee.write("\n")
            tee.flush()
            self.assertEqual(json.loads(log.getvalue().splitlines()[-1])["generation"], 1)
            self.assertEqual(tee.trace_steps, 1)
            self.assertEqual(original.getvalue(), log.getvalue())

        def test_no_trace_is_not_reported_as_success(self):
            with tempfile.TemporaryDirectory() as folder:
                args = self.make_args(Path(folder))
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, "No steering trace"):
                        run_experiment(args, lambda a, d: self.factory(a, d, with_trace=False))

        def test_validation_rejects_ambiguous_conditions(self):
            for sentences, emotions, strengths, names in (
                (SENTENCES[:4], EMOTIONS, (1.0, 2.0), EMOTIONS),
                (SENTENCES, ("happy", "happy"), (1.0,), EMOTIONS),
                (SENTENCES, EMOTIONS, (0.0,), EMOTIONS),
                (SENTENCES, EMOTIONS, (float("nan"),), EMOTIONS),
                (SENTENCES, EMOTIONS, (1.0, 1.0), EMOTIONS),
                (SENTENCES, EMOTIONS, (1.0,), ("happy", "angry")),
            ):
                with self.assertRaises(ValueError):
                    build_plan(sentences, emotions, strengths, names)

        def test_random_seed_is_rejected(self):
            with tempfile.TemporaryDirectory() as folder:
                args = self.make_args(Path(folder))
                args.seed = -1
                with self.assertRaisesRegex(ValueError, "seed"):
                    run_experiment(args, self.factory)

        def test_repeated_runs_do_not_overwrite(self):
            with tempfile.TemporaryDirectory() as folder:
                args = self.make_args(Path(folder))
                with contextlib.redirect_stderr(io.StringIO()):
                    first = run_experiment(args, self.factory)
                    original = (first / "steering_manifest.json").read_bytes()
                    second = run_experiment(args, self.factory)
                self.assertNotEqual(first, second)
                self.assertEqual((first / "steering_manifest.json").read_bytes(), original)

        def test_wave_writer_rejects_non_pcm16(self):
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaisesRegex(ValueError, "PCM16"):
                    write_wave(Path(folder) / "wrong.wav", 24000, array("f", [0.1]))

        def test_mock_backend_does_not_import_torch_or_tts(self):
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("TTS_infer_pack.TTS", sys.modules)

        def test_webui_t4_and_cpu_defaults(self):
            from unittest.mock import patch
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(webui_runtime_defaults(True), {"device": "cuda", "is_half": True})
                self.assertEqual(webui_runtime_defaults(False), {"device": "cpu", "is_half": False})
                self.assertEqual(webui_runtime_defaults(True, "cpu"), {"device": "cpu", "is_half": False})
            with patch.dict(os.environ, {"is_half": "False"}):
                self.assertFalse(webui_runtime_defaults(True)["is_half"])

        def test_mismatched_trace_runtime_is_rejected(self):
            fixed = {"top_k": 15, "top_p": 1., "temperature": 1., "repetition_penalty": 1.35}
            runtime = {"probability_dtype": "torch.float32", "score_dtype": "torch.float32",
                       "device_type": "cpu", "sampling": fixed}
            check_trace_runtime(runtime, {"dtype": "torch.float32", "device": "cpu"}, fixed)
            for mismatch in ({"probability_dtype": "torch.float16"}, {"score_dtype": None},
                             {"device_type": "cuda"}, {"sampling": {}}):
                with self.assertRaisesRegex(ValueError, "Trace runtime differs"):
                    check_trace_runtime({**runtime, **mismatch}, {"dtype": "torch.float32", "device": "cpu"}, fixed)

        def test_factory_uses_existing_api_and_isolates_config_writes(self):
            from types import ModuleType
            from unittest.mock import patch
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                args = self.make_args(root)
                source = {"custom": {"version": "v2", "device": "cpu"}}
                Path(args.config).write_text(json.dumps(source), encoding="utf-8")
                original = Path(args.config).read_bytes()
                run_dir = root / "run"
                run_dir.mkdir()
                metadata = {"emotion_names": list(EMOTIONS), "extra": {
                    "version": "v5dev", "s1_sha256": file_hash(args.s1_checkpoint),
                }}
                modules = {name: ModuleType(name) for name in (
                    "AR", "AR.models", "AR.models.emotion_control",
                    "TTS_infer_pack", "TTS_infer_pack.TTS", "yaml", "torch",
                )}
                modules["AR.models.emotion_control"].STEERING_TRACE = True
                modules["AR.models.emotion_control"].load_emotion_controller_checkpoint = (
                    lambda *a, **kw: (None, metadata)
                )
                modules["yaml"].safe_load = json.loads
                from types import SimpleNamespace
                modules["torch"].cuda = SimpleNamespace(is_available=lambda: True)
                seen = {}

                class Config:
                    def __init__(_, data):
                        seen["config"] = data
                        _.languages = ["all_ko"]

                class Pipeline:
                    def __init__(_, config):
                        seen["save_path"] = config.configs_path
                        Path(config.configs_path).write_text("runtime-only", encoding="utf-8")

                    def load_emotion_controller(_, path):
                        seen["controller"] = path
                        return metadata

                modules["TTS_infer_pack.TTS"].TTS_Config = Config
                modules["TTS_infer_pack.TTS"].TTS = Pipeline
                modules["TTS_infer_pack.TTS"].set_seed = lambda seed: seed
                previous_dir = Path.cwd()
                try:
                    with patch.dict(sys.modules, modules), patch.object(sys, "path", list(sys.path)), \
                            patch.dict(os.environ, {"is_half": "True"}):
                        _, actual_metadata, _ = create_pipeline(args, run_dir)
                finally:
                    os.chdir(previous_dir)
                self.assertEqual(actual_metadata, metadata)
                self.assertEqual(seen["save_path"], str(run_dir / "tts_runtime.yaml"))
                self.assertEqual(seen["config"]["custom"]["version"], "v5dev")
                self.assertEqual(seen["config"]["custom"]["device"], "cuda")
                self.assertTrue(seen["config"]["custom"]["is_half"])
                self.assertEqual(seen["config"]["custom"]["t2s_weights_path"], args.s1_checkpoint)
                self.assertEqual(seen["config"]["custom"]["vits_weights_path"], args.s2_checkpoint)
                self.assertEqual(Path(args.config).read_bytes(), original)

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(RunnerTests)
    )
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_tests()
    if args.dry_run:
        if not 0 <= args.seed < 2**32:
            parser.error("A fixed seed in 0..2**32-1 is required")
        sentences = SENTENCES if not args.sentences_json else json.loads(
            Path(args.sentences_json).read_text(encoding="utf-8")
        )
        plan = build_plan(sentences, args.emotions, args.strengths, args.emotions, args.limit)
        print(json.dumps({
            "mode": "dry_run", "checkpoint_pair_and_emotion_ids": "not_verified",
            "requests": [{**{k: v for k, v in row.items() if k != "emotion_id"},
                          "seed": args.seed} for row in plan],
        }, ensure_ascii=False, indent=2))
        return 0
    directory = run_experiment(args)
    print(f"RUN_DIR={directory}")
    print(f"Manifest: {directory / 'steering_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
