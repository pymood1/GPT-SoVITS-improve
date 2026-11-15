"""Run the V5 acoustic A/B diagnostic end-to-end in Google Colab."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile


BRANCH = "codex/v5-acoustic-ab"
REPOSITORY = "https://github.com/pymood1/GPT-SoVITS-improve.git"
HF_REPOSITORY = "lj1995/GPT-SoVITS"
COLAB_REPO = Path("/content/GPT-SoVITS-acoustic-ab")
COLAB_OUTPUT = Path("/content/v5_acoustic_ab")
REFERENCE_URL = (
    "https://download.pytorch.org/torchaudio/tutorial-assets/"
    "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
)
REFERENCE_SHA256 = "c65fcd726d6b08c82c1e5dc7558f863cd8d483e3ed2f4a7bcf271dc1865ada14"
REFERENCE_TEXT = "I had that curiosity beside me at this moment."
TEST_TEXT = "안녕하세요. 오늘은 날씨가 좋아서 잠깐 산책을 다녀오려고 합니다."
MODEL_PATTERNS = (
    "s1v3.ckpt",
    "gsv-v5-pretrained/s2Gv5dev.pth",
    "gsv-v5-pretrained/vocoder.pth",
    "chinese-hubert-base/*",
    "chinese-roberta-wwm-ext-large/*",
)
KOREAN_REFERENCE_LISTS = (
    "/content/audio_input/training.list",
    "/content/GPT-SoVITS/training.list",
    "/content/drive/MyDrive/audio_input/training.list",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_download(url: str, destination: Path, expected_sha256: str) -> Path:
    """Download once and reject a partial or replaced public reference file."""
    destination = Path(destination)
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, temporary)
        actual = _sha256(temporary)
        if actual != expected_sha256:
            raise RuntimeError(
                f"참조 음성 SHA-256 불일치: expected={expected_sha256}, actual={actual}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def runtime_environment(
    repo: Path,
    output: Path,
    reference: Path,
    reference_text: str = REFERENCE_TEXT,
    reference_language: str = "en",
) -> dict[str, str]:
    repo, output, reference = map(Path, (repo, output, reference))
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
        "GPT_SOVITS_AB_REFERENCE_TEXT": reference_text,
        "GPT_SOVITS_AB_REFERENCE_LANGUAGE": reference_language,
        "GPT_SOVITS_AB_TARGET_TEXT": TEST_TEXT,
        "PYTHONUNBUFFERED": "1",
    }


def ensure_nltk_frontend_resources() -> None:
    """Prepare only missing English frontend data and verify the real POS path."""
    import nltk

    resources = (
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/cmudict", "cmudict"),
    )
    for lookup, package in resources:
        try:
            nltk.data.find(lookup)
        except LookupError:
            print(f"NLTK frontend resource missing; installing once: {package}", flush=True)
            nltk.download(package, quiet=True)
            nltk.data.find(lookup)

    # Verify the operation that failed in Colab, not merely downloader metadata.
    try:
        nltk.pos_tag(["frontend", "check"])
    except LookupError:
        # Older NLTK releases can still request the legacy tagger name.
        print("NLTK legacy POS tagger is also required; installing once.", flush=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.pos_tag(["frontend", "check"])

    # g2p_en additionally consumes cmudict; instantiate and execute a tiny probe.
    from g2p_en import G2p

    probe = G2p()
    if not probe("frontend check"):
        raise RuntimeError("g2p_en frontend self-check returned no phones")
    print("NLTK/g2p_en frontend self-check: OK", flush=True)


def select_korean_reference(list_paths, info):
    """Prefer an existing 3–10 s Korean training reference when one is available."""
    for raw_list in list_paths:
        if not raw_list:
            continue
        list_path = Path(raw_list)
        if not list_path.is_file():
            continue
        candidates = []
        for row in list_path.read_text(encoding="utf-8-sig").splitlines()[:1000]:
            fields = row.split("|", 3)
            if len(fields) != 4:
                continue
            raw_path, _, language, text = fields
            if language.strip().lower() not in {"ko", "ko_kr", "ko-kr", "korean"} or not text.strip():
                continue
            name = raw_path.replace("\\", "/").rsplit("/", 1)[-1]
            paths = (
                Path(raw_path),
                list_path.parent / raw_path,
                list_path.parent / name,
                list_path.parent / "datas" / name,
                Path("/content/audio_input/datas") / name,
                Path("/content/drive/MyDrive/audio_input/datas") / name,
            )
            for path in paths:
                if not path.is_file():
                    continue
                try:
                    duration = info(str(path)).duration
                except Exception:
                    continue
                if 3 <= duration <= 10:
                    candidates.append((abs(duration - 6), path.resolve(), text.strip()))
                    break
        if candidates:
            _, path, text = min(candidates)
            return path, text
    return None


def _language_label(mapping: dict[str, str], value: str) -> str:
    try:
        return next(label for label, mapped in mapping.items() if mapped == value)
    except StopIteration as exc:
        raise RuntimeError(f"언어 매핑에서 {value!r}을 찾지 못했습니다.") from exc


def run_headless(
    webui,
    reference: str,
    reference_text: str = REFERENCE_TEXT,
    reference_language: str = "en",
) -> int:
    """Drive the normal semantic path once; its opt-in hook writes the A/B suite."""
    prompt_language = _language_label(webui.dict_language, reference_language)
    target_language = _language_label(webui.dict_language, "all_ko")
    generated = webui.get_tts_wav(
        reference,
        reference_text,
        prompt_language,
        TEST_TEXT,
        target_language,
        sample_steps=32,
        cfg_rate=1.30,
        use_cuda_graph=False,
        use_flash_attention=False,
    )
    count = 0
    for _ in generated:
        count += 1
    if count == 0:
        raise RuntimeError("일반 semantic 경로가 오디오를 한 번도 생성하지 않았습니다.")
    return count


def find_completed_suite(output: Path, not_before_ns: int = 0) -> Path:
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


def _run(
    command: list[str], *, cwd: Path | None = None, env=None, log_path: Path | None = None
) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    tail = deque(maxlen=200)
    log = None
    try:
        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
        for line in process.stdout:
            print(line, end="", flush=True)
            tail.append(line)
            if log is not None:
                log.write(line)
                log.flush()
    finally:
        process.stdout.close()
        if log is not None:
            log.close()
    return_code = process.wait()
    if return_code:
        location = f"\n전체 자식 로그: {log_path}" if log_path is not None else ""
        raise RuntimeError(
            f"자식 프로세스 실패 (exit code {return_code}).{location}\n"
            f"마지막 출력:\n{''.join(tail)}"
        )


def _prepare_checkout(repo: Path) -> None:
    if not (repo / ".git").is_dir():
        _run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(repo)])
        return
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True
    )
    if dirty.strip():
        raise RuntimeError(f"진단 checkout의 기존 수정사항을 보존하기 위해 중단했습니다: {repo}")
    _run(["git", "fetch", "origin", BRANCH], cwd=repo)
    _run(["git", "merge", "--ff-only", "FETCH_HEAD"], cwd=repo)


def _install_runtime(repo: Path) -> None:
    if shutil.which("ffmpeg") is None:
        _run(["apt-get", "update"])
        _run(["apt-get", "install", "-y", "ffmpeg", "libsox-dev"])
    _run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=repo)
    ensure_nltk_frontend_resources()


def _download_models(repo: Path) -> None:
    from huggingface_hub import snapshot_download

    pretrained = repo / "GPT_SoVITS/pretrained_models"
    print("V5dev pretrained 모델을 준비합니다. 첫 실행은 다운로드에 시간이 걸립니다.", flush=True)
    snapshot_download(
        repo_id=HF_REPOSITORY,
        allow_patterns=list(MODEL_PATTERNS),
        local_dir=str(pretrained),
    )
    required = [
        pretrained / "s1v3.ckpt",
        pretrained / "gsv-v5-pretrained/s2Gv5dev.pth",
        pretrained / "gsv-v5-pretrained/vocoder.pth",
        pretrained / "chinese-hubert-base",
        pretrained / "chinese-roberta-wwm-ext-large",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("pretrained 파일 다운로드가 완전하지 않습니다: " + ", ".join(missing))


def _listening_bundle(folder: Path) -> Path:
    archive = Path("/content/v5_acoustic_ab_listening.zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(folder / "report.json", "report.json")
        for path in sorted(folder.rglob("*_listen.wav")):
            bundle.write(path, str(path.relative_to(folder)))
    return archive


def _display_results(reference: Path, folder: Path) -> None:
    from IPython.display import Audio, FileLink, display

    print("\n0. 공개 참조 원본", flush=True)
    display(Audio(filename=str(reference)))
    sys.path.insert(0, str(COLAB_REPO))
    from tools.kaggle_v5_acoustic_ab import show_report

    show_report(folder)
    archive = _listening_bundle(folder)
    print("청취 파일 + report.json 다운로드:", flush=True)
    display(FileLink(str(archive)))


def _headless_main(repo: Path, output: Path, reference: Path) -> None:
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo / "GPT_SoVITS")]
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Colab 런타임 유형을 GPU로 설정한 뒤 Run all을 다시 실행하세요.")
    import inference_webui as webui

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("자동 합성 문장:", TEST_TEXT, flush=True)
    reference_text = os.environ.get("GPT_SOVITS_AB_REFERENCE_TEXT", REFERENCE_TEXT)
    reference_language = os.environ.get("GPT_SOVITS_AB_REFERENCE_LANGUAGE", "en")
    print("참조 언어:", reference_language, flush=True)
    run_headless(webui, str(reference), reference_text, reference_language)
    folder = find_completed_suite(output)
    print(f"COLAB_V5_AB_RESULT={folder}", flush=True)


def _running_from_colab_checkout() -> bool:
    """Return false for raw ``exec`` bootstraps, which do not define __file__."""
    source = globals().get("__file__")
    if not source:
        return False
    return Path(source).resolve().parent == (COLAB_REPO / "tools").resolve()


def run_colab_all() -> Path:
    """Public notebook entry point: setup, one synthesis, controls, playback."""
    if not Path("/content").is_dir():
        raise RuntimeError("이 자동 실행기는 Google Colab에서 사용하세요.")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Colab 메뉴에서 런타임 유형을 GPU로 설정한 뒤 Run all을 누르세요.")
    _prepare_checkout(COLAB_REPO)
    # Re-exec the just-fetched implementation so an old raw bootstrap cannot run stale logic.
    if not _running_from_colab_checkout():
        import importlib.util

        current = COLAB_REPO / "tools/colab_v5_acoustic_ab.py"
        spec = importlib.util.spec_from_file_location("colab_v5_acoustic_ab_current", current)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.run_colab_all()
    _install_runtime(COLAB_REPO)
    _download_models(COLAB_REPO)
    COLAB_OUTPUT.mkdir(parents=True, exist_ok=True)
    import soundfile

    candidate_lists = (
        os.environ.get("GPT_SOVITS_DEFAULT_TRAINING_LIST", ""),
        *KOREAN_REFERENCE_LISTS,
    )
    korean_reference = select_korean_reference(candidate_lists, soundfile.info)
    if korean_reference is not None:
        reference, reference_text = korean_reference
        reference_language = "all_ko"
        print("기존 한국어 reference를 우선 사용합니다:", reference, flush=True)
    else:
        reference = verified_download(
            REFERENCE_URL, COLAB_OUTPUT / "public_reference.wav", REFERENCE_SHA256
        )
        reference_text = REFERENCE_TEXT
        reference_language = "en"
        print("한국어 reference를 찾지 못해 검증된 공개 영어 reference를 사용합니다.", flush=True)
    started = time.time_ns()
    env = dict(os.environ)
    env.pop("emotion_controller_path", None)
    env.update(
        runtime_environment(
            COLAB_REPO,
            COLAB_OUTPUT,
            reference,
            reference_text=reference_text,
            reference_language=reference_language,
        )
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [str(COLAB_REPO), str(COLAB_REPO / "GPT_SoVITS"), env.get("PYTHONPATH", "")]
    )
    _run(
        [
            sys.executable,
            "-u",
            "tools/colab_v5_acoustic_ab.py",
            "--headless",
            "--reference",
            str(reference),
            "--output",
            str(COLAB_OUTPUT),
        ],
        cwd=COLAB_REPO,
        env=env,
        log_path=COLAB_OUTPUT / "headless.log",
    )
    folder = find_completed_suite(COLAB_OUTPUT, started)
    _display_results(reference, folder)
    return folder


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, default=COLAB_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.headless:
        if args.reference is None:
            raise SystemExit("--reference is required with --headless")
        _headless_main(Path(__file__).resolve().parents[1], args.output, args.reference)
    else:
        run_colab_all()
