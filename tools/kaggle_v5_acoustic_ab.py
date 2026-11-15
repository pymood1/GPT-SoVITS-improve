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
DEFAULT_SOURCE = Path("/kaggle/tmp/GPT-SoVITS")
DEFAULT_OUTPUT = Path("/kaggle/working/v5_acoustic_ab")


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
        Path("/kaggle/input/datasets/pymood1"),
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

    repo = Path(repo or Path(__file__).resolve().parents[1])
    source = DEFAULT_SOURCE
    output = DEFAULT_OUTPUT
    if not source.is_dir():
        raise RuntimeError(
            "기존 GPT-SoVITS 설치 /kaggle/tmp/GPT-SoVITS가 없습니다. "
            "평소 설치/학습 셀을 먼저 실행하세요."
        )

    link_assets(source, repo)
    required = [
        repo / "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        repo / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth",
        repo / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth",
        repo / "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        repo / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("기존 설치에서 pretrained 자산을 찾지 못했습니다: " + ", ".join(missing))

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
