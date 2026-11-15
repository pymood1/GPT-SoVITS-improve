"""Headless Kaggle V5 acoustic A/B using the user's existing runtime and data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

BRANCH = "codex/v5-acoustic-ab"
REPOSITORY = "https://github.com/pymood1/GPT-SoVITS-improve.git"
TEST_TEXT = "안녕하세요. 오늘은 날씨가 좋아서 잠깐 산책을 다녀오려고 합니다."
DEFAULT_OUTPUT = Path("/kaggle/working/v5_acoustic_ab")
HF_REPOSITORY = "lj1995/GPT-SoVITS"
MODEL_PATTERNS = (
    "s1v3.ckpt",
    "gsv-v5-pretrained/s2Gv5dev.pth",
    "gsv-v5-pretrained/vocoder.pth",
    "chinese-hubert-base/*",
    "chinese-roberta-wwm-ext-large/*",
)


def _required_pretrained(root):
    pretrained = Path(root) / "GPT_SoVITS/pretrained_models"
    return [
        pretrained / "s1v3.ckpt",
        pretrained / "gsv-v5-pretrained/s2Gv5dev.pth",
        pretrained / "gsv-v5-pretrained/vocoder.pth",
        pretrained / "chinese-hubert-base",
        pretrained / "chinese-roberta-wwm-ext-large",
    ]


def _has_pretrained(root):
    return all(path.exists() for path in _required_pretrained(root))


def discover_existing_install(exclude=None):
    """Find a usable prior checkout instead of assuming one Kaggle path."""
    exclude = Path(exclude).resolve() if exclude else None
    explicit = os.environ.get("GPT_SOVITS_ROOT", "").strip()
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            Path("/kaggle/tmp/GPT-SoVITS"),
            Path("/kaggle/working/GPT-SoVITS"),
            Path("/kaggle/tmp/gpt_sovits_all_in_one/GPT-SoVITS"),
            Path("/kaggle/working/gpt_sovits_all_in_one/GPT-SoVITS"),
            Path("/kaggle/tmp/gpt_sovits_all_in_one"),
            Path("/kaggle/working/gpt_sovits_all_in_one"),
        ]
    )
    for root in (Path("/kaggle/tmp"), Path("/kaggle/working")):
        if not root.is_dir():
            continue
        try:
            for pretrained in root.glob("**/GPT_SoVITS/pretrained_models"):
                candidates.append(pretrained.parents[1])
        except OSError:
            pass

    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        key = str(resolved)
        if key in seen or (exclude is not None and resolved == exclude):
            continue
        seen.add(key)
        if (resolved / "GPT_SoVITS").is_dir() and _has_pretrained(resolved):
            return resolved
    return None


def ensure_runtime(repo, reuse_source=None):
    """Prefer the existing environment; install repo requirements only as fallback."""
    probe = [
        sys.executable,
        "-c",
        "import torch, torchaudio, transformers, peft, librosa, soundfile, x_transformers",
    ]
    if subprocess.run(probe, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return
    print("필수 Python 패키지가 부족해 Git checkout의 requirements를 설치합니다.", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        cwd=repo,
        check=True,
    )


def ensure_pretrained(repo, reuse_source=None):
    """Reuse existing pretrained assets or download only the missing public base assets."""
    repo = Path(repo)
    if _has_pretrained(repo):
        print("진단 checkout의 기존 pretrained 자산을 재사용합니다.", flush=True)
        return
    if reuse_source is not None and _has_pretrained(reuse_source):
        print("기존 GPT-SoVITS 설치의 pretrained 자산을 재사용합니다:", reuse_source, flush=True)
        link_assets(reuse_source, repo)
        if _has_pretrained(repo):
            return

    print("기존 pretrained 설치를 찾지 못해 기본 V5dev 자산만 자동 준비합니다.", flush=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.32,<1", "hf_xet>=1.7"],
            check=True,
        )
        from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPOSITORY,
        allow_patterns=list(MODEL_PATTERNS),
        local_dir=str(repo / "GPT_SoVITS/pretrained_models"),
    )
    missing = [str(path) for path in _required_pretrained(repo) if not path.exists()]
    if missing:
        raise RuntimeError("pretrained 자동 준비가 불완전합니다: " + ", ".join(missing))


def link_assets(source, target):
    """Reuse the already-installed runtime/pretrained assets without copying them."""
    pairs = [
        (source / "runtime", target / "runtime"),
        (source / "GPT_SoVITS/text/G2PWModel", target / "GPT_SoVITS/text/G2PWModel"),
    ]
    pretrained = source / "GPT_SoVITS/pretrained_models"
    if pretrained.is_dir():
        pairs.extend(
            (p, target / "GPT_SoVITS/pretrained_models" / p.name)
            for p in pretrained.iterdir()
            if p.name != ".gitignore"
        )
    for original, destination in pairs:
        if original.exists() and not destination.exists() and not destination.is_symlink():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(original.resolve(), target_is_directory=original.is_dir())


def discover_training_lists():
    """Find existing GPT-SoVITS metadata; never invent or download a reference."""
    candidates = []
    explicit = os.environ.get("GPT_SOVITS_DEFAULT_TRAINING_LIST", "").strip()
    if explicit:
        candidates.append(Path(explicit))

    fixed = [
        Path("/kaggle/tmp/audio_input/training.list"),
        Path("/kaggle/tmp/audio_input/asr_opt.list"),
        Path("/kaggle/working/audio_input/training.list"),
        Path("/kaggle/working/training.list"),
        Path("/kaggle/input/datasets/pymood1/endfield/training.list"),
    ]
    candidates.extend(fixed)

    # Existing training pipelines use several temp/workspace layouts. Search only
    # Kaggle-local roots and cap results so dataset discovery cannot become costly.
    for root in (
        Path("/kaggle/tmp"),
        Path("/kaggle/working"),
        Path("/kaggle/input"),
    ):
        if not root.is_dir():
            continue
        found = 0
        try:
            for path in root.rglob("*.list"):
                candidates.append(path)
                found += 1
                if found >= 200:
                    break
        except OSError:
            pass

    unique = []
    seen = set()
    for path in candidates:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _candidate_audio_paths(list_path, raw_path):
    raw = Path(raw_path)
    name = raw_path.replace("\\", "/").rsplit("/", 1)[-1]
    return [
        raw,
        list_path.parent / raw,
        list_path.parent / name,
        list_path.parent / "datas" / name,
        Path("/kaggle/tmp/audio_input/datas") / name,
        Path("/kaggle/working/audio_input/datas") / name,
    ]


def select_reference(lists, info):
    """Pick an existing Korean 3-10 s clip and preserve its exact transcript."""
    accepted = {"ko", "ko_kr", "ko-kr", "korean", "all_ko"}
    for list_path in lists:
        list_path = Path(list_path)
        if not list_path.is_file():
            continue
        candidates = []
        try:
            rows = list_path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError):
            continue
        for row in rows[:2000]:
            fields = row.split("|", 3)
            if len(fields) != 4 or fields[2].strip().lower() not in accepted:
                continue
            raw_path, _, _, text = fields
            text = text.strip()
            if not text:
                continue
            for path in _candidate_audio_paths(list_path, raw_path):
                if not path.is_file():
                    continue
                try:
                    duration = float(info(str(path)).duration)
                except Exception:
                    continue
                if 3 <= duration <= 10:
                    candidates.append((abs(duration - 6), str(path.resolve()), text))
                    break
        if candidates:
            _, audio, text = min(candidates)
            return audio, text, str(list_path.resolve())
    return "", "", ""


def find_completed_suite(output, not_before_ns=0):
    candidates = []
    for report_path in Path(output).glob("*/report.json"):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            modified = report_path.stat().st_mtime_ns
        except (OSError, ValueError):
            continue
        if (
            payload.get("suite")
            and payload.get("completed") is True
            and not payload.get("failure")
            and modified >= not_before_ns
        ):
            candidates.append((modified, report_path.parent))
    if not candidates:
        raise RuntimeError("완료된 V5 acoustic A/B report.json을 찾지 못했습니다.")
    return max(candidates, key=lambda item: item[0])[1]


def show_report(folder):
    """Display the exact existing reference and the diagnostic listening controls."""
    from IPython.display import Audio, display

    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    preferred = [
        ("cold_B", "reference_roundtrip"),
        ("cold_A", "webui_A"),
        ("cold_B", "official_B"),
        ("fp32", "official_B"),
        ("fp32", "reference_roundtrip"),
        ("forward", "webui_A_repeat"),
        ("reverse", "official_B"),
        ("reverse", "webui_A"),
    ]
    labels = [
        "1. 참조 mel → 현재 precision vocoder (CFM 제외)",
        "2. 새 인스턴스 WebUI A",
        "3. 새 인스턴스 공식 B",
        "4. 동일 semantic/초기 noise · 완전 FP32 공식 B",
        "5. 참조 mel → FP32 vocoder (CFM 제외)",
        "6. 정방향 A-repeat",
        "7. 역방향 B",
        "8. 역방향 A",
    ]
    for (sequence, route), label in zip(preferred, labels):
        options = [
            entry
            for entry in report.get("listening_files", [])
            if entry["sequence"] == sequence and entry["route"] == route
        ]
        if not options:
            continue
        entry = next((e for e in options if e.get("case") == "full"), options[0])
        print(label)
        display(Audio(filename=str(folder / entry["path"])))
    print("report.json:", folder / "report.json")
    for case, comparisons in report.get("comparisons", {}).items():
        print(
            case,
            {
                key: value.get("first_difference")
                for key, value in comparisons.items()
                if isinstance(value, dict)
            },
        )


def _listening_bundle(folder):
    archive = DEFAULT_OUTPUT / "latest_listening.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(Path(folder) / "report.json", "report.json")
        for path in sorted(Path(folder).rglob("*_listen.wav")):
            bundle.write(path, str(path.relative_to(folder)))
    return archive


def _runtime_environment(repo, output, reference, transcript):
    pretrained = repo / "GPT_SoVITS/pretrained_models"
    return {
        "language": "ko_KR",
        "version": "v5dev",
        "is_half": "True",
        "gpt_path": str(pretrained / "s1v3.ckpt"),
        "sovits_path": str(pretrained / "gsv-v5-pretrained/s2Gv5dev.pth"),
        "bert_path": str(pretrained / "chinese-roberta-wwm-ext-large"),
        "cnhubert_base_path": str(pretrained / "chinese-hubert-base"),
        "GPT_SOVITS_V5_ACOUSTIC_AB_DIR": str(output),
        "GPT_SOVITS_V5_AB_REQUIRE_PRETRAINED": "1",
        "GPT_SOVITS_AB_REFERENCE": str(reference),
        "GPT_SOVITS_AB_REFERENCE_TEXT": transcript,
        "GPT_SOVITS_AB_REFERENCE_LANGUAGE": "all_ko",
        "GPT_SOVITS_AB_TARGET_TEXT": TEST_TEXT,
        "PYTHONUNBUFFERED": "1",
    }


def run_headless(webui, reference, transcript):
    prompt_language = next(
        label for label, mapped in webui.dict_language.items() if mapped == "all_ko"
    )
    target_language = prompt_language
    generated = webui.get_tts_wav(
        reference,
        transcript,
        prompt_language,
        TEST_TEXT,
        target_language,
        sample_steps=32,
        cfg_rate=1.30,
        use_cuda_graph=False,
        use_flash_attention=False,
    )
    count = sum(1 for _ in generated)
    if count == 0:
        raise RuntimeError("일반 semantic 경로가 오디오를 생성하지 않았습니다.")
    return count


def _headless_main(repo, output, reference, transcript):
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo / "GPT_SoVITS")]
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU를 활성화하세요.")
    import inference_webui as webui

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("기존 데이터 reference:", reference, flush=True)
    print("기존 데이터 전사:", transcript, flush=True)
    print("목표 문장:", TEST_TEXT, flush=True)
    run_headless(webui, str(reference), transcript)
    folder = find_completed_suite(output)
    print(f"KAGGLE_V5_AB_RESULT={folder}", flush=True)


def run_kaggle_all(repo=None):
    """Reuse an existing Kaggle install/data and run the full diagnostic headlessly."""
    import soundfile
    import torch

    if not Path("/kaggle").is_dir():
        raise RuntimeError("Kaggle 세션에서 실행하세요.")
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU를 활성화하세요.")

    repo = Path(repo or Path(__file__).resolve().parents[1]).resolve()
    output = DEFAULT_OUTPUT

    source = discover_existing_install(exclude=repo)
    if source is not None:
        print("기존 GPT-SoVITS 설치 자동 탐지:", source, flush=True)
        link_assets(source, repo)
    else:
        print("기존 GPT-SoVITS 설치 checkout은 찾지 못했습니다. Git 진단 checkout을 직접 사용합니다.", flush=True)

    ensure_runtime(repo, source)
    ensure_pretrained(repo, source)

    lists = discover_training_lists()
    reference, transcript, used_list = select_reference(lists, soundfile.info)
    if not reference:
        inspected = "\n".join(f"- {path}" for path in lists[:30]) or "- 발견된 .list 없음"
        raise RuntimeError(
            "기존 데이터에서 3~10초 한국어 reference + 정확한 전사를 찾지 못했습니다. "
            "공개/임의 샘플로 fallback하지 않습니다.\n확인한 목록:\n" + inspected
        )

    from IPython.display import Audio, display

    print("사용할 기존 training list:", used_list, flush=True)
    print("사용할 기존 reference:", reference, flush=True)
    print("전사:", transcript, flush=True)
    display(Audio(filename=reference))

    output.mkdir(parents=True, exist_ok=True)
    started = time.time_ns()
    env = dict(os.environ)
    env.pop("emotion_controller_path", None)
    env.update(_runtime_environment(repo, output, reference, transcript))
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo), str(repo / "GPT_SoVITS"), env.get("PYTHONPATH", "")]
    )

    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--headless",
        "--repo",
        str(repo),
        "--output",
        str(output),
        "--reference",
        reference,
        "--transcript",
        transcript,
    ]
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=repo, env=env, text=True)
    if completed.returncode:
        raise RuntimeError(f"Kaggle V5 acoustic A/B 실패 (exit code {completed.returncode})")

    folder = find_completed_suite(output, started)
    show_report(folder)
    archive = _listening_bundle(folder)
    print("청취 WAV + report.json:", archive, flush=True)
    return folder


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference")
    parser.add_argument("--transcript")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.headless:
        if not args.repo or not args.reference or not args.transcript:
            raise SystemExit("--repo, --reference, --transcript are required with --headless")
        _headless_main(args.repo, args.output, args.reference, args.transcript)
    else:
        run_kaggle_all()
