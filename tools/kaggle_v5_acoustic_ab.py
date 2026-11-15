"""One-cell Kaggle launcher; reuse installed packages/assets, preserve user source."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time

BRANCH = "codex/v5-acoustic-ab"
REPOSITORY = "https://github.com/pymood1/GPT-SoVITS-improve.git"
TEST_TEXT = "안녕하세요. 오늘은 날씨가 좋아서 잠깐 산책을 다녀오려고 합니다."


def link_assets(source, target):
    """Link only reusable runtime assets, never replace existing target files."""
    pairs = [(source / "runtime", target / "runtime"),
             (source / "GPT_SoVITS/text/G2PWModel", target / "GPT_SoVITS/text/G2PWModel")]
    pretrained = source / "GPT_SoVITS/pretrained_models"
    if pretrained.is_dir():
        pairs.extend((p, target / "GPT_SoVITS/pretrained_models" / p.name)
                     for p in pretrained.iterdir() if p.name != ".gitignore")
    for original, destination in pairs:
        if original.exists() and not destination.exists() and not destination.is_symlink():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(original.resolve(), target_is_directory=original.is_dir())


def select_reference(lists, info):
    """Select an existing Korean 3–10s reference and its exact .list transcript."""
    for list_path in lists:
        list_path = Path(list_path)
        if not list_path.is_file():
            continue
        candidates = []
        for row in list_path.read_text(encoding="utf-8-sig").splitlines()[:500]:
            fields = row.split("|", 3)
            if len(fields) != 4 or fields[2].strip().lower() not in {"ko", "ko_kr", "korean"}:
                continue
            raw_path, _, _, text = fields
            if not text.strip():
                continue
            name = raw_path.replace("\\", "/").rsplit("/", 1)[-1]
            paths = [Path(raw_path), list_path.parent / name, list_path.parent / "datas" / name,
                     Path("/kaggle/tmp/audio_input/datas") / name]
            for path in paths:
                if not path.is_file():
                    continue
                try:
                    duration = info(str(path)).duration
                except Exception:
                    continue
                if 3 <= duration <= 10:
                    candidates.append((abs(duration - 6), str(path.resolve()), text.strip()))
                    break
        if candidates:
            _, audio, text = min(candidates)
            return audio, text
    return "", ""


def show_report(folder):
    """Notebook-local playback only; no result or private-input upload."""
    from IPython.display import Audio, display
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if report.get("failure"):
        print("진단 실패:", report["failure"], "보고서:", folder / "report.json")
        return
    preferred = [("cold_B", "reference_roundtrip"), ("cold_A", "webui_A"),
                 ("cold_B", "official_B"), ("fp32", "official_B"),
                 ("fp32", "reference_roundtrip"),
                 ("forward", "webui_A_repeat"), ("reverse", "official_B"),
                 ("reverse", "webui_A")]
    labels = ["1. 참조 mel → 현재 precision vocoder (CFM 제외)",
              "2. 새 인스턴스 WebUI A", "3. 새 인스턴스 공식 B",
              "4. 동일 semantic/초기 noise · 완전 FP32 공식 B",
              "5. 참조 mel → FP32 vocoder (CFM 제외)",
              "6. 정방향 A-repeat", "7. 역방향 B", "8. 역방향 A"]
    for (sequence, route), label in zip(preferred, labels):
        options = [entry for entry in report["listening_files"]
                   if entry["sequence"] == sequence and entry["route"] == route]
        if not options:
            continue
        entry = next((e for e in options if e.get("case") == "full"), options[0])
        print(label)
        display(Audio(filename=str(folder / entry["path"])))
    print("report.json:", folder / "report.json")
    for case, comparisons in report["comparisons"].items():
        print(case, {k: v["first_difference"] for k, v in comparisons.items() if isinstance(v, dict)})
    print("아직 원인 확정이 아닙니다. 1→2→3→4 순서로 노이즈를 비교하세요.")


def launch():
    if not Path("/kaggle").is_dir():
        raise RuntimeError("이 셀은 기존 GPT-SoVITS 설치가 있는 Kaggle GPU 세션에서 실행하세요.")
    source = Path("/kaggle/tmp/GPT-SoVITS")
    target = Path("/kaggle/tmp/GPT-SoVITS-acoustic-ab")
    output = Path("/kaggle/working/v5_acoustic_ab")
    marker = output / "launcher-session.json"
    if marker.is_file():
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
            pid = int(previous["pid"])
            command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().replace("\0", " ")
            if "GPT_SoVITS/inference_webui.py" in command and Path(f"/proc/{pid}/cwd").resolve() == target.resolve():
                old_log = Path(previous["log"]).read_text(encoding="utf-8")[-65536:]
                links = re.findall(r"https://[A-Za-z0-9-]+\.gradio\.live", old_log)
                print("진단 WebUI가 이미 실행 중입니다:", links[-1] if links else "공유 링크 준비 중")
                return None
        except (OSError, ValueError, KeyError):
            pass
    if not source.is_dir():
        raise RuntimeError("기존 설치 /kaggle/tmp/GPT-SoVITS가 없습니다. 평소 설치 셀을 먼저 실행하세요.")
    if not (target / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                        REPOSITORY, str(target)], check=True)
    else:
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=target, text=True)
        if dirty.strip():
            raise RuntimeError("진단 전용 checkout에 수정된 파일이 있어 보존했습니다: " + str(target))
        subprocess.run(["git", "fetch", "origin", BRANCH], cwd=target, check=True)
        subprocess.run(["git", "merge", "--ff-only", "FETCH_HEAD"], cwd=target, check=True)
    # Import the updated checked-out helper; bootstrap code might be from an older cell.
    link_assets(source, target)
    required = ["GPT_SoVITS/pretrained_models/s1v3.ckpt",
                "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth",
                "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth",
                "GPT_SoVITS/pretrained_models/chinese-hubert-base",
                "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"]
    missing = [p for p in required if not (target / p).exists()]
    if missing:
        raise RuntimeError("기존 설치에서 필요한 pretrained 파일을 찾지 못했습니다: " + ", ".join(missing))
    subprocess.run([sys.executable, "-c",
                    "import torch; assert torch.cuda.is_available(), 'Kaggle GPU를 활성화하세요'; print('GPU:', torch.cuda.get_device_name(0))"], check=True)
    import soundfile
    lists = [os.environ.get("GPT_SOVITS_DEFAULT_TRAINING_LIST", ""),
             "/kaggle/tmp/audio_input/training.list",
             "/kaggle/input/datasets/pymood1/endfield/training.list"]
    reference, transcript = select_reference(lists, soundfile.info)
    if not reference:
        print("자동 선택할 한국어 reference가 없습니다. WebUI에서 3–10초 음성과 해당 전사문만 입력하세요.")
    else:
        from IPython.display import Audio, display
        print("자동 선택한 참조 원본입니다. 이 원본에 이미 노이즈가 있는지도 먼저 확인하세요.")
        display(Audio(filename=reference))
    output.mkdir(parents=True, exist_ok=True)
    seen = {str(p) for p in output.glob("*/report.json")}
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, language="ko_KR", version="v5dev", is_share="True",
               gpt_path=required[0], sovits_path=required[1],
               infer_ttswebui=str(port), GPT_SOVITS_V5_ACOUSTIC_AB_DIR=str(output),
               GPT_SOVITS_V5_AB_REQUIRE_PRETRAINED="1",
               GPT_SOVITS_AB_REFERENCE=reference, GPT_SOVITS_AB_REFERENCE_TEXT=transcript,
               GPT_SOVITS_AB_TARGET_TEXT=TEST_TEXT, PYTHONUNBUFFERED="1")
    env.pop("emotion_controller_path", None)
    # Preserve existing CUDA/PyTorch/dependency environment; no install/reset/kill.
    env["PYTHONPATH"] = os.pathsep.join([str(target), str(target / "GPT_SoVITS"), env.get("PYTHONPATH", "")])
    log_path = output / f"launcher-{time.time_ns()}.log"
    process = subprocess.Popen([sys.executable, "-u", "GPT_SoVITS/inference_webui.py", "ko_KR"],
                               cwd=target, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    marker.write_text(json.dumps({"pid": process.pid, "log": str(log_path)}), encoding="utf-8")
    def logs():
        printed = set()
        with log_path.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                for url in re.findall(r"https://[A-Za-z0-9-]+\.gradio\.live", line):
                    if url not in printed:
                        print("진단 WebUI:", url, "\n기본 모델/한국어 문장이 준비되어 있습니다. 합성 버튼을 한 번 누르세요.", flush=True)
                        printed.add(url)
                if "[V5 acoustic" in line:
                    print(line.strip(), flush=True)
        if process.wait():
            print("WebUI 종료. 전체 로그:", log_path, flush=True)
    def watch():
        while process.poll() is None:
            for path in output.glob("*/report.json"):
                if str(path) in seen:
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if payload.get("suite"):
                    seen.add(str(path))
                    show_report(path.parent)
            time.sleep(2)
    threading.Thread(target=logs, daemon=True).start()
    threading.Thread(target=watch, daemon=True).start()
    print("공유 링크를 준비 중입니다. 이 셀 출력에 링크와 비교용 플레이어가 자동으로 표시됩니다.")
    return process


if __name__ == "__main__":
    launch()
