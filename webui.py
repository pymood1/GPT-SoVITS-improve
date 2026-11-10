import os
import sys
import re
import threading
import time
from collections import deque
from pathlib import Path

version = os.environ.get("version", "v2Pro")
os.environ["version"] = version
now_dir = os.getcwd()
sys.path.insert(0, now_dir)
import warnings

warnings.filterwarnings("ignore")
import json
import platform
import shutil
import signal

import psutil
import torch
import yaml

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
torch.manual_seed(233333)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
if os.path.exists(tmp):
    for name in os.listdir(tmp):
        if name == "jieba.cache":
            continue
        path = "%s/%s" % (tmp, name)
        delete = os.remove if os.path.isfile(path) else shutil.rmtree
        try:
            delete(path)
        except Exception as e:
            print(str(e))
            pass
import site
import traceback

site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if site_packages_roots == []:
    site_packages_roots = ["%s/runtime/Lib/site-packages" % now_dir]
# os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["all_proxy"] = ""
for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/users.pth" % (site_packages_root), "w") as f:
                f.write(
                    # "%s\n%s/runtime\n%s/tools\n%s/tools/asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    "%s\n%s/GPT_SoVITS/BigVGAN\n%s/tools\n%s/tools/asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    % (now_dir, now_dir, now_dir, now_dir, now_dir, now_dir)
                )
            break
        except PermissionError:
            traceback.print_exc()
import shutil
import subprocess
from subprocess import Popen

from tools.assets import css, js, top_html
from tools.i18n.i18n import I18nAuto, scan_language_list

language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else "Auto"
os.environ["language"] = language
i18n = I18nAuto(language=language)
from multiprocessing import cpu_count

from config import (
    GPU_INDEX,
    GPU_INFOS,
    IS_GPU,
    exp_root,
    infer_device,
    is_half,
    is_share,
    memset,
    python_exec,
    webui_port_infer_tts,
    webui_port_main,
    webui_port_subfix,
    webui_port_uvr5,
)
from tools import my_utils
from tools.emotion_training_pipeline import (
    EmotionPipelineError,
    create_reference_map,
    detect_pair_emotions,
    format_dataset_summary,
    latest_feature_build,
    validate_emotion_dataset,
    write_v5_tts_config,
)
from tools.my_utils import check_details, check_for_existance

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # 当遇到mps不支持的步骤时使用cpu
import gradio as gr

n_cpu = cpu_count()

set_gpu_numbers = GPU_INDEX
gpu_infos = GPU_INFOS
mem = memset
is_gpu_ok = IS_GPU

v3v4set = {"v3", "v4", "v5dev", "v5turbo"}


def set_default():
    global \
        default_batch_size, \
        default_max_batch_size, \
        gpu_info, \
        default_sovits_epoch, \
        default_sovits_save_every_epoch, \
        max_sovits_epoch, \
        max_sovits_save_every_epoch, \
        default_batch_size_s1, \
        if_force_ckpt
    if_force_ckpt = False
    gpu_info = "\n".join(gpu_infos)
    if is_gpu_ok:
        minmem = min(mem)
        default_batch_size = int(minmem // 2 if version not in v3v4set else minmem // 8)
        default_batch_size_s1 = int(minmem // 2)
    else:
        default_batch_size = default_batch_size_s1 = int(psutil.virtual_memory().total / 1024 / 1024 / 1024 / 4)
    if version not in v3v4set:
        default_sovits_epoch = 8
        default_sovits_save_every_epoch = 4
        max_sovits_epoch = 25  # 40
        max_sovits_save_every_epoch = 25  # 10
    else:
        default_sovits_epoch = 2
        default_sovits_save_every_epoch = 1
        max_sovits_epoch = 16  # 40 # 3 #训太多=作死
        max_sovits_save_every_epoch = 10  # 10 # 3

    default_batch_size = max(1, default_batch_size)
    default_batch_size_s1 = max(1, default_batch_size_s1)
    default_max_batch_size = default_batch_size * 3


set_default()

gpus = "-".join(map(str, GPU_INDEX))
default_gpu_numbers = infer_device.index


def fix_gpu_number(input):  # 将越界的number强制改到界内
    try:
        if int(input) not in set_gpu_numbers:
            return default_gpu_numbers
    except:
        return input
    return input


def fix_gpu_numbers(inputs):
    output = []
    try:
        for input in inputs.split(","):
            output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs


from config import (
    name2gpt_path,
    name2sovits_path,
    pretrained_gpt_name,
    pretrained_sovits_name,
)


def check_pretrained_is_exist(version):
    pretrained_model_list = (
        pretrained_sovits_name[version],
        pretrained_gpt_name[version],
        "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    )
    if version not in v3v4set:
        pretrained_model_list += (pretrained_sovits_name[version].replace("s2G", "s2D"),)
    _ = ""
    for i in pretrained_model_list:
        if os.path.exists(i) == False:
            _ += f"\n    {i}"
    if _:
        print("warning: ", i18n("以下模型不存在:") + _)


check_pretrained_is_exist(version)
for key in pretrained_sovits_name.keys():
    if os.path.exists(pretrained_sovits_name[key]) == False:
        pretrained_sovits_name[key] = ""
for key in pretrained_gpt_name.keys():
    if os.path.exists(pretrained_gpt_name[key]) == False:
        pretrained_gpt_name[key] = ""

from config import (
    GPT_weight_root,
    GPT_weight_version2root,
    SoVITS_weight_root,
    SoVITS_weight_version2root,
    change_choices,
    get_weights_names,
)

for root in SoVITS_weight_root + GPT_weight_root:
    os.makedirs(root, exist_ok=True)
SoVITS_names, GPT_names = get_weights_names()


def _default_pretrained_choice(names, version_):
    preferred = i18n(f"不训练直接推{version_}底模！")
    if preferred in names:
        return preferred
    return names[0] if names else ""


default_sovits_choice = _default_pretrained_choice(SoVITS_names, version)
default_gpt_choice = _default_pretrained_choice(GPT_names, version)

p_label = None
p_uvr5 = None
p_asr = None
asr_cancel_requested = False
p_denoise = None
p_tts_inference = None
p_emotion_training = None
latest_emotion_controller_path = os.environ.get(
    "emotion_controller_path",
    "",
)
tts_inference_url = None
tts_inference_logs = deque(maxlen=40)
tts_inference_reader = None


def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass


system = platform.system()


def kill_process(pid, process_name=""):
    if system == "Windows":
        cmd = "taskkill /t /f /pid %s" % pid
        # os.system(cmd)
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        kill_proc_tree(pid)
    print(process_name + i18n("进程已终止"))


def process_info(process_name="", indicator=""):
    if indicator == "opened":
        return process_name + i18n("已开启")
    elif indicator == "open":
        return i18n("开启") + process_name
    elif indicator == "closed":
        return process_name + i18n("已关闭")
    elif indicator == "close":
        return i18n("关闭") + process_name
    elif indicator == "running":
        return process_name + i18n("运行中")
    elif indicator == "occupy":
        return process_name + i18n("占用中") + "," + i18n("需先终止才能开启下一次任务")
    elif indicator == "finish":
        return process_name + i18n("已完成")
    elif indicator == "failed":
        return process_name + i18n("失败")
    elif indicator == "info":
        return process_name + i18n("进程输出信息")
    else:
        return process_name



def _emotion_status(stage, log_lines, final_message=None):
    tail = list(log_lines)[-28:]
    message = f"### {stage}"

    if final_message:
        message += f"\n\n{final_message}"

    if tail:
        message += "\n\n```text\n"
        message += "\n".join(tail)
        message += "\n```"

    return message


def _stream_emotion_command(cmd, log_lines, stage):
    global p_emotion_training

    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"

    command_text = " ".join(str(item) for item in cmd)
    log_lines.append(f"$ {command_text}")
    print(f"[Emotion] {command_text}", flush=True)

    process = Popen(
        cmd,
        cwd=now_dir,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    p_emotion_training = process

    if process.stdout is not None:
        for line in process.stdout:
            clean_line = line.rstrip()
            log_lines.append(clean_line)
            print(f"[Emotion] {clean_line}", flush=True)
            yield _emotion_status(stage, log_lines)

    return_code = process.wait()
    p_emotion_training = None

    if return_code != 0:
        raise EmotionPipelineError(
            f"{stage} failed with exit code {return_code}"
        )


def stop_emotion_training():
    global p_emotion_training

    if p_emotion_training is None:
        return "### 감정 작업\n\n실행 중인 작업이 없습니다."

    if p_emotion_training.poll() is None:
        kill_process(
            p_emotion_training.pid,
            "Emotion task ",
        )

    p_emotion_training = None

    return "### 감정 작업 중지됨"


def run_auto_emotion_sidecar(
    source_list,
    wav_root,
    sidecar_output,
    model_name,
    min_confidence,
    device,
):
    global p_emotion_training

    if p_emotion_training is not None and p_emotion_training.poll() is None:
        yield (
            "### 자동 감정 데이터 생성\n\n"
            "이미 다른 감정 작업이 실행 중입니다.",
            {"__type__": "update"},
        )
        return

    log_lines = deque(maxlen=120)

    try:
        source_list = str(source_list or "").strip()
        wav_root = str(wav_root or "").strip()
        model_name = str(model_name or "").strip()
        device = str(device or "auto").strip() or "auto"

        if not source_list:
            raise EmotionPipelineError(
                "먼저 GPT-SoVITS 텍스트 표기(.list) 파일을 지정하세요."
            )

        source_path = Path(source_list).expanduser()

        if not sidecar_output:
            sidecar_output = str(
                source_path.with_suffix(
                    source_path.suffix + ".emotion.auto.jsonl"
                )
            )

        output_path = Path(
            str(sidecar_output)
        ).expanduser().resolve()

        if not model_name:
            model_name = "iic/emotion2vec_plus_base"

        min_confidence = float(min_confidence)

        if not 0.0 <= min_confidence <= 1.0:
            raise EmotionPipelineError(
                "감정 신뢰도 기준은 0~1 사이여야 합니다."
            )

        cmd = [
            python_exec,
            "-u",
            "tools/auto_emotion_sidecar.py",
            "--list",
            source_list,
            "--output",
            str(output_path),
            "--model",
            model_name,
            "--device",
            device,
            "--min-confidence",
            str(min_confidence),
        ]

        if wav_root:
            cmd.extend(
                [
                    "--wav-root",
                    wav_root,
                ]
            )

        for status in _stream_emotion_command(
            cmd,
            log_lines,
            "음성 감정 분석 및 pair 생성",
        ):
            yield (
                status,
                {"__type__": "update"},
            )

        if not output_path.is_file():
            raise EmotionPipelineError(
                f"자동 sidecar 파일이 생성되지 않았습니다: {output_path}"
            )

        report_path = output_path.with_suffix(
            output_path.suffix + ".report.json"
        )
        summary = None

        if report_path.is_file():
            report = json.loads(
                report_path.read_text(encoding="utf-8")
            )
            summary = (
                f"샘플 {report.get('samples', 0)}개, "
                f"pair group {report.get('pair_groups', 0)}개, "
                f"pair 샘플 {report.get('paired_samples', 0)}개"
            )

        yield (
            _emotion_status(
                "자동 감정 데이터 생성 완료",
                log_lines,
                (
                    f"**Sidecar:** `{output_path}`"
                    + (f"\n\n{summary}" if summary else "")
                ),
            ),
            {
                "__type__": "update",
                "value": str(output_path),
            },
        )

    except (
        EmotionPipelineError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        p_emotion_training = None
        log_lines.append(f"[ERROR] {exc}")
        yield (
            _emotion_status(
                "자동 감정 데이터 생성 실패",
                log_lines,
                str(exc),
            ),
            {"__type__": "update"},
        )


def run_emotion_training_pipeline(
    source_list,
    wav_root,
    sidecar_path,
    dataset_id,
    dataset_dir,
    feature_root,
    controller_output,
    s1_checkpoint_path,
    emotion_names,
    steps,
    gpu_number,
    reuse_dataset,
):
    global p_emotion_training
    global latest_emotion_controller_path

    if p_emotion_training is not None and p_emotion_training.poll() is None:
        yield "### 감정 제어 학습\n\n이미 다른 감정 학습 작업이 실행 중입니다."
        return

    log_lines = deque(maxlen=120)

    try:
        dataset_id = str(dataset_id or "").strip() or "emotion_dataset_01"
        dataset_slug = re.sub(
            r"[^0-9A-Za-z._-]+",
            "_",
            dataset_id,
        ).strip("._-") or "emotion_dataset_01"

        dataset_dir_text = str(dataset_dir or "").strip()
        if not dataset_dir_text:
            dataset_dir_text = f"/content/{dataset_slug}_emotion_dataset"

        feature_root_text = str(feature_root or "").strip()
        if not feature_root_text:
            feature_root_text = f"/content/{dataset_slug}_emotion_features"

        controller_output_text = str(
            controller_output or ""
        ).strip()
        if not controller_output_text:
            controller_output_text = (
                "/content/drive/MyDrive/GPT-SoVITS/"
                f"{dataset_slug}_emotion_controller.pt"
            )

        dataset_dir = Path(dataset_dir_text).expanduser().resolve()
        feature_root = Path(feature_root_text).expanduser().resolve()
        controller_output = Path(
            controller_output_text
        ).expanduser().resolve()

        metadata_path = dataset_dir / "metadata.jsonl"
        train_list = dataset_dir / "train.list"
        reference_map_path = dataset_dir / "reference_map.json"
        reference_prompt_path = dataset_dir / "reference_prompts.pt"
        tts_config_path = dataset_dir / "tts_infer_emotion_v5dev.yaml"

        s1_checkpoint_path = str(
            s1_checkpoint_path or ""
        ).strip()

        if not s1_checkpoint_path:
            s1_checkpoint_path = (
                "GPT_SoVITS/pretrained_models/s1v3.ckpt"
            )

        s1_checkpoint_path = name2gpt_path.get(
            s1_checkpoint_path,
            s1_checkpoint_path,
        )

        s1_checkpoint = Path(
            s1_checkpoint_path
        ).expanduser().resolve()

        s2_checkpoint = Path(
            "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/"
            "s2Gv5dev.pth"
        ).resolve()
        bert_dir = Path(
            "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        ).resolve()
        cnhubert_dir = Path(
            "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        ).resolve()

        for required_path, label in (
            (s1_checkpoint, "V5dev S1"),
            (s2_checkpoint, "V5dev S2"),
            (bert_dir, "BERT"),
            (cnhubert_dir, "CNHuBERT"),
        ):
            if not required_path.exists():
                raise EmotionPipelineError(
                    f"{label} path not found: {required_path}"
                )

        requested_emotions = [
            item.strip()
            for item in str(emotion_names).split(",")
            if item.strip()
        ]

        if len(set(requested_emotions)) != len(requested_emotions):
            raise EmotionPipelineError(
                "감정 이름이 중복되어 있습니다."
            )

        if requested_emotions and len(requested_emotions) < 2:
            raise EmotionPipelineError(
                "감정을 직접 지정할 경우 최소 2개가 필요합니다."
            )

        steps = int(steps)

        if steps <= 0:
            raise EmotionPipelineError(
                "학습 step은 1 이상이어야 합니다."
            )

        gpu_number = str(fix_gpu_number(gpu_number))

        dataset_ready = (
            metadata_path.is_file()
            and train_list.is_file()
            and (dataset_dir / "report.json").is_file()
        )

        if reuse_dataset and dataset_ready:
            log_lines.append(
                f"[reuse] canonical dataset: {dataset_dir}"
            )
            yield _emotion_status(
                "1/5 감정 데이터셋 검사",
                log_lines,
            )
        else:
            source_list = str(source_list or "").strip()
            sidecar_path = str(sidecar_path or "").strip()
            wav_root = str(wav_root or "").strip()

            if not source_list:
                raise EmotionPipelineError(
                    "원본 GPT-SoVITS .list 파일 경로를 입력하세요."
                )

            if not sidecar_path:
                raise EmotionPipelineError(
                    "감정 pair 학습에는 sidecar JSONL이 필요합니다."
                )

            if dataset_dir.exists() and any(dataset_dir.iterdir()):
                raise EmotionPipelineError(
                    "canonical dataset 폴더가 이미 비어 있지 않습니다. "
                    "'기존 canonical dataset 재사용'을 켜거나 "
                    "다른 출력 경로를 사용하세요."
                )

            cmd = [
                python_exec,
                "-u",
                "tools/emotion_dataset_builder.py",
                "--list",
                source_list,
                "--output",
                str(dataset_dir),
                "--dataset-id",
                dataset_id,
                "--sidecar",
                sidecar_path,
                "--stage-mode",
                "symlink",
            ]

            if wav_root:
                cmd.extend(
                    [
                        "--wav-root",
                        wav_root,
                    ]
                )

            for status in _stream_emotion_command(
                cmd,
                log_lines,
                "1/5 감정 데이터셋 생성",
            ):
                yield status

        report = validate_emotion_dataset(dataset_dir)
        summary = format_dataset_summary(report)
        log_lines.append(f"[dataset] {summary}")

        detected_emotions = detect_pair_emotions(metadata_path)

        if requested_emotions:
            missing_emotions = [
                emotion
                for emotion in requested_emotions
                if emotion not in detected_emotions
            ]
            if missing_emotions:
                raise EmotionPipelineError(
                    "사용 가능한 pair 데이터가 없는 감정: "
                    + ", ".join(missing_emotions)
                )
            parsed_emotions = requested_emotions
        else:
            parsed_emotions = detected_emotions
            log_lines.append(
                "[emotions] auto-detected: "
                + ",".join(parsed_emotions)
            )

        create_reference_map(
            metadata_path,
            reference_map_path,
        )
        log_lines.append(
            f"[reference] {reference_map_path}"
        )

        yield _emotion_status(
            "2/5 기준 reference 자동 선택",
            log_lines,
            "3~10초 train 샘플 중 emotion pair 밖의 reference를 선택했습니다.",
        )

        preprocess_cmd = [
            python_exec,
            "-u",
            "tools/run_emotion_v5_preprocess.py",
            "--train-list",
            str(train_list),
            "--output-root",
            str(feature_root),
            "--bert-dir",
            str(bert_dir),
            "--cnhubert-dir",
            str(cnhubert_dir),
            "--pretrained-s2g",
            str(s2_checkpoint),
            "--version",
            "v5dev",
            "--gpu",
            gpu_number,
        ]

        for status in _stream_emotion_command(
            preprocess_cmd,
            log_lines,
            "3/5 V5dev 전처리",
        ):
            yield status

        feature_dir = None

        for line in reversed(log_lines):
            candidate_text = str(line).strip()

            if not candidate_text:
                continue

            candidate = Path(candidate_text).expanduser()

            if (
                candidate.name.startswith("build_")
                and candidate.is_dir()
                and (candidate / "build.json").is_file()
            ):
                feature_dir = candidate.resolve()
                break

        if feature_dir is None:
            feature_dir = latest_feature_build(feature_root)

        build_manifest = feature_dir / "build.json"

        log_lines.append(
            f"[features] {feature_dir}"
        )

        write_v5_tts_config(
            tts_config_path,
            version="v5dev",
            s1_checkpoint=s1_checkpoint,
            s2_checkpoint=s2_checkpoint,
            bert_dir=bert_dir,
            cnhubert_dir=cnhubert_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
            is_half=bool(is_half and torch.cuda.is_available()),
        )

        prompt_cmd = [
            python_exec,
            "-u",
            "tools/export_emotion_reference_prompts.py",
            "--tts-config",
            str(tts_config_path),
            "--metadata",
            str(metadata_path),
            "--reference-map",
            str(reference_map_path),
            "--output",
            str(reference_prompt_path),
        ]

        for status in _stream_emotion_command(
            prompt_cmd,
            log_lines,
            "4/5 reference prompt 생성",
        ):
            yield status

        controller_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_cmd = [
            python_exec,
            "-u",
            "tools/train_emotion_controller.py",
            "--s1-checkpoint",
            str(s1_checkpoint),
            "--feature-dir",
            str(feature_dir),
            "--feature-build-manifest",
            str(build_manifest),
            "--metadata",
            str(metadata_path),
            "--reference-map",
            str(reference_map_path),
            "--reference-prompt-cache",
            str(reference_prompt_path),
            "--output",
            str(controller_output),
            "--emotions",
            ",".join(parsed_emotions),
            "--steps",
            str(steps),
            "--device",
            "cuda" if torch.cuda.is_available() else "cpu",
        ]

        for status in _stream_emotion_command(
            train_cmd,
            log_lines,
            "5/5 emotion controller 학습",
        ):
            yield status

        if not controller_output.is_file():
            raise EmotionPipelineError(
                "학습 프로세스는 종료됐지만 controller 파일이 없습니다: "
                f"{controller_output}"
            )

        latest_emotion_controller_path = str(controller_output)
        os.environ["emotion_controller_path"] = (
            latest_emotion_controller_path
        )

        yield _emotion_status(
            "감정 제어 학습 완료",
            log_lines,
            (
                f"**Controller:** `{controller_output}`\n\n"
                "이 파일을 1C-추론의 감정 컨트롤러에서 불러오면 됩니다."
            ),
        )

    except (
        EmotionPipelineError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        p_emotion_training = None
        log_lines.append(f"[ERROR] {exc}")
        yield _emotion_status(
            "감정 제어 학습 실패",
            log_lines,
            str(exc),
        )


process_name_subfix = i18n("音频标注WebUI")


def change_label(path_list):
    global p_label
    if p_label is None:
        check_for_existance([path_list])
        path_list = my_utils.clean_path(path_list)
        cmd = '"%s" -s tools/subfix_webui.py --load_list "%s" --webui_port %s --is_share %s' % (
            python_exec,
            path_list,
            webui_port_subfix,
            is_share,
        )
        yield (
            process_info(process_name_subfix, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        print(cmd)
        p_label = Popen(cmd, shell=True)
    else:
        kill_process(p_label.pid, process_name_subfix)
        p_label = None
        yield (
            process_info(process_name_subfix, "closed"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )


process_name_uvr5 = i18n("人声分离WebUI")


def change_uvr5():
    global p_uvr5
    if p_uvr5 is None:
        cmd = '"%s" -s tools/uvr5/webui.py "%s" %s %s %s' % (
            python_exec,
            infer_device,
            is_half,
            webui_port_uvr5,
            is_share,
        )
        yield (
            process_info(process_name_uvr5, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        print(cmd)
        p_uvr5 = Popen(cmd, shell=True)
    else:
        kill_process(p_uvr5.pid, process_name_uvr5)
        p_uvr5 = None
        yield (
            process_info(process_name_uvr5, "closed"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )


process_name_tts = i18n("TTS推理WebUI")


def _stream_tts_inference_output(process):
    global tts_inference_url

    if process.stdout is None:
        return

    public_url_pattern = re.compile(r"https://[A-Za-z0-9-]+\.gradio\.live")

    for line in process.stdout:
        clean_line = line.rstrip()
        print(f"[TTS] {clean_line}", flush=True)
        tts_inference_logs.append(clean_line)

        match = public_url_pattern.search(clean_line)
        if match is not None:
            tts_inference_url = match.group(0)


def _tts_log_tail(limit=12):
    lines = list(tts_inference_logs)[-limit:]
    if not lines:
        return ""
    return "\n".join(lines)


def change_tts_inference(
    bert_path,
    cnhubert_base_path,
    gpu_number,
    gpt_path,
    sovits_path,
    batched_infer_enabled,
):
    global p_tts_inference
    global tts_inference_reader
    global tts_inference_url

    if p_tts_inference is not None and p_tts_inference.poll() is not None:
        p_tts_inference = None

    if p_tts_inference is None:
        gpt_path = name2gpt_path.get(gpt_path, gpt_path)
        sovits_path = name2sovits_path.get(sovits_path, sovits_path)

        controller_available = bool(
            latest_emotion_controller_path
            and os.path.isfile(latest_emotion_controller_path)
        )

        script_path = (
            "GPT_SoVITS/inference_webui.py"
            if controller_available
            else (
                "GPT_SoVITS/inference_webui_fast.py"
                if batched_infer_enabled
                else "GPT_SoVITS/inference_webui.py"
            )
        )

        cmd = [
            python_exec,
            "-u",
            "-s",
            script_path,
            language,
        ]

        child_env = os.environ.copy()
        child_env.update(
            {
                "gpt_path": gpt_path,
                "sovits_path": sovits_path,
                "cnhubert_base_path": cnhubert_base_path,
                "bert_path": bert_path,
                "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_number)),
                "is_half": str(is_half),
                "infer_ttswebui": str(webui_port_infer_tts),
                "is_share": str(is_share),
                "version": version,
                "language": language,
                "emotion_controller_path": latest_emotion_controller_path,
                "PYTHONUNBUFFERED": "1",
            }
        )

        tts_inference_url = None
        tts_inference_logs.clear()

        print("[TTS] launch:", " ".join(cmd), flush=True)

        process = Popen(
            cmd,
            cwd=now_dir,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        p_tts_inference = process

        tts_inference_reader = threading.Thread(
            target=_stream_tts_inference_output,
            args=(process,),
            daemon=True,
        )
        tts_inference_reader.start()

        yield (
            i18n("正在启动TTS推理WebUI，请等待模型加载和公开链接生成。"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )

        deadline = time.monotonic() + 120

        while time.monotonic() < deadline:
            if tts_inference_url is not None:
                yield (
                    f"### {i18n('TTS推理WebUI已就绪')}\n\n"
                    f"[{i18n('TTS推理WebUI')}]({tts_inference_url})",
                    {"__type__": "update", "visible": False},
                    {"__type__": "update", "visible": True},
                )
                return

            return_code = process.poll()
            if return_code is not None:
                log_tail = _tts_log_tail()
                p_tts_inference = None

                message = (
                    f"### {i18n('TTS推理WebUI启动失败')} "
                    f"(exit code {return_code})"
                )

                if log_tail:
                    message += f"\n\n```text\n{log_tail}\n```"

                yield (
                    message,
                    {"__type__": "update", "visible": True},
                    {"__type__": "update", "visible": False},
                )
                return

            time.sleep(0.25)

        log_tail = _tts_log_tail()
        message = f"### {i18n('未能生成公开链接，请查看日志')}"

        if log_tail:
            message += f"\n\n```text\n{log_tail}\n```"

        yield (
            message,
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )

    else:
        kill_process(p_tts_inference.pid, process_name_tts)
        p_tts_inference = None
        tts_inference_url = None
        tts_inference_logs.clear()

        yield (
            process_info(process_name_tts, "closed"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )


from tools.asr.config import asr_dict

ASR_UI_CHOICES = [
    ("Faster Whisper (다국어, 권장)", "Faster Whisper (多语种)"),
    ("Fun-ASR MLT-Nano (다국어)", "Fun-ASR MLT-Nano (多语种)"),
    ("SenseVoice (5개 언어, 빠름)", "SenseVoice (极速, 5语种)"),
    ("Damo ASR (중국어)", "达摩 ASR (中文经典)"),
]
DEFAULT_ASR_MODEL = "Faster Whisper (多语种)"
DEFAULT_ASR_SIZE = "large-v3-turbo"
DEFAULT_ASR_LANGUAGE = "auto"
DEFAULT_ASR_PRECISION = "float16" if is_half else "float32"

process_name_asr = i18n("语音识别")


def open_asr(
    asr_inp_dir,
    asr_opt_dir,
    asr_model,
    asr_model_size,
    asr_lang,
    asr_precision,
):
    global p_asr
    global asr_cancel_requested

    if p_asr is not None:
        yield (
            process_info(process_name_asr, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    asr_inp_dir = str(asr_inp_dir or "").strip().strip("'"")
    asr_opt_dir = str(asr_opt_dir or "").strip().strip("'"")

    if not check_for_existance([asr_inp_dir]):
        yield (
            process_info(process_name_asr, "failed"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    asr_config = asr_dict.get(asr_model)
    if asr_config is None:
        yield (
            process_info(process_name_asr, "failed")
            + f": unknown ASR model {asr_model!r}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    cmd = [
        python_exec,
        "-s",
        f"tools/asr/{asr_config['path']}",
        "-i",
        asr_inp_dir,
        "-o",
        asr_opt_dir,
        "-s",
        str(asr_model_size),
        "-l",
        str(asr_lang),
        "-p",
        str(asr_precision),
    ]

    backend = asr_config.get("backend")
    if backend:
        cmd.extend(["--backend", str(backend)])

    input_name = os.path.basename(os.path.normpath(asr_inp_dir))
    output_root = Path(asr_opt_dir or "output/asr_opt").expanduser()
    if not output_root.is_absolute():
        output_root = Path(now_dir) / output_root
    output_file_path = (output_root / f"{input_name}.list").resolve()

    yield (
        process_info(process_name_asr, "opened"),
        {"__type__": "update", "visible": False},
        {"__type__": "update", "visible": True},
        {"__type__": "update"},
        {"__type__": "update"},
        {"__type__": "update"},
    )

    print("[ASR]", " ".join(repr(item) for item in cmd), flush=True)
    asr_cancel_requested = False

    try:
        p_asr = Popen(
            cmd,
            cwd=now_dir,
        )
        return_code = p_asr.wait()
    except OSError as exc:
        p_asr = None
        yield (
            process_info(process_name_asr, "failed") + f": {exc}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return
    finally:
        p_asr = None

    if asr_cancel_requested:
        asr_cancel_requested = False
        return

    if return_code != 0 or not output_file_path.is_file():
        yield (
            (
                process_info(process_name_asr, "failed")
                + f" (exit code {return_code})"
            ),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    yield (
        process_info(process_name_asr, "finish"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
        {"__type__": "update", "value": str(output_file_path)},
        {"__type__": "update", "value": str(output_file_path)},
        {"__type__": "update", "value": asr_inp_dir},
    )


def close_asr():
    global p_asr
    global asr_cancel_requested

    if p_asr is not None:
        asr_cancel_requested = True
        kill_process(p_asr.pid, process_name_asr)
        p_asr = None

    return (
        process_info(process_name_asr, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


process_name_denoise = i18n("语音降噪")


def open_denoise(denoise_inp_dir, denoise_opt_dir):
    global p_denoise
    if p_denoise == None:
        denoise_inp_dir = my_utils.clean_path(denoise_inp_dir)
        denoise_opt_dir = my_utils.clean_path(denoise_opt_dir)
        check_for_existance([denoise_inp_dir])
        cmd = '"%s" -s tools/cmd-denoise.py -i "%s" -o "%s" -p %s' % (
            python_exec,
            denoise_inp_dir,
            denoise_opt_dir,
            "float16" if is_half == True else "float32",
        )

        yield (
            process_info(process_name_denoise, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        print(cmd)
        p_denoise = Popen(cmd, shell=True)
        p_denoise.wait()
        p_denoise = None
        yield (
            process_info(process_name_denoise, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update", "value": denoise_opt_dir},
            {"__type__": "update", "value": denoise_opt_dir},
        )
    else:
        yield (
            process_info(process_name_denoise, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
        )


def close_denoise():
    global p_denoise
    if p_denoise is not None:
        kill_process(p_denoise.pid, process_name_denoise)
        p_denoise = None
    return (
        process_info(process_name_denoise, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


p_train_SoVITS = None
process_name_sovits = i18n("SoVITS训练")


def open1Ba(
    version,
    batch_size,
    total_epoch,
    exp_name,
    text_low_lr_rate,
    if_save_latest,
    if_save_every_weights,
    save_every_epoch,
    gpu_numbers1Ba,
    pretrained_s2G,
    pretrained_s2D,
    if_grad_ckpt,
    lora_rank,
):
    global p_train_SoVITS
    if p_train_SoVITS == None:
        exp_name = exp_name.rstrip(" ")
        config_file = (
            "GPT_SoVITS/configs/s2.json"
            if version not in {"v2Pro", "v2ProPlus"}
            else f"GPT_SoVITS/configs/s2{version}.json"
        )
        with open(config_file) as f:
            data = f.read()
            data = json.loads(data)
        s2_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s2_%s" % (s2_dir, version), exist_ok=True)
        if check_for_existance([s2_dir], is_train=True):
            check_details([s2_dir], is_train=True)
        if is_half == False:
            data["train"]["fp16_run"] = False
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["train"]["text_low_lr_rate"] = text_low_lr_rate
        data["train"]["pretrained_s2G"] = pretrained_s2G
        data["train"]["pretrained_s2D"] = pretrained_s2D
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["save_every_epoch"] = save_every_epoch
        data["train"]["gpu_numbers"] = gpu_numbers1Ba
        data["train"]["grad_ckpt"] = if_grad_ckpt
        data["train"]["lora_rank"] = lora_rank
        data["model"]["version"] = version
        data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
        data["save_weight_dir"] = SoVITS_weight_version2root[version]
        data["name"] = exp_name
        data["version"] = version
        tmp_config_path = "%s/tmp_s2.json" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(json.dumps(data))
        if version in ["v1", "v2", "v2Pro", "v2ProPlus"]:
            cmd = '"%s" -s GPT_SoVITS/s2_train.py --config "%s"' % (python_exec, tmp_config_path)
        else:
            cmd = '"%s" -s GPT_SoVITS/s2_train_v3_lora.py --config "%s"' % (python_exec, tmp_config_path)
        yield (
            process_info(process_name_sovits, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        print(cmd)
        p_train_SoVITS = Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        p_train_SoVITS = None
        SoVITS_dropdown_update, GPT_dropdown_update = change_choices()
        yield (
            process_info(process_name_sovits, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            SoVITS_dropdown_update,
            GPT_dropdown_update,
        )
    else:
        yield (
            process_info(process_name_sovits, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
        )


def close1Ba():
    global p_train_SoVITS
    if p_train_SoVITS is not None:
        kill_process(p_train_SoVITS.pid, process_name_sovits)
        p_train_SoVITS = None
    return (
        process_info(process_name_sovits, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


p_train_GPT = None
process_name_gpt = i18n("GPT训练")


def open1Bb(
    batch_size,
    total_epoch,
    exp_name,
    if_dpo,
    if_save_latest,
    if_save_every_weights,
    save_every_epoch,
    gpu_numbers,
    pretrained_s1,
):
    global p_train_GPT
    if p_train_GPT == None:
        exp_name = exp_name.rstrip(" ")
        with open(
            "GPT_SoVITS/configs/s1longer.yaml" if version == "v1" else "GPT_SoVITS/configs/s1longer-v2.yaml"
        ) as f:
            data = f.read()
            data = yaml.load(data, Loader=yaml.FullLoader)
        s1_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s1" % (s1_dir), exist_ok=True)
        if check_for_existance([s1_dir], is_train=True):
            check_details([s1_dir], is_train=True)
        if is_half == False:
            data["train"]["precision"] = "32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["pretrained_s1"] = pretrained_s1
        data["train"]["save_every_n_epoch"] = save_every_epoch
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_dpo"] = if_dpo
        data["train"]["half_weights_save_dir"] = GPT_weight_version2root[version]
        data["train"]["exp_name"] = exp_name
        data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
        data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
        data["output_dir"] = "%s/logs_s1_%s" % (s1_dir, version)
        # data["version"]=version

        os.environ["_CUDA_VISIBLE_DEVICES"] = str(fix_gpu_numbers(gpu_numbers.replace("-", ",")))
        os.environ["hz"] = "25hz"
        tmp_config_path = "%s/tmp_s1.yaml" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        cmd = '"%s" -s GPT_SoVITS/s1_train.py --config_file "%s" ' % (python_exec, tmp_config_path)
        yield (
            process_info(process_name_gpt, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        print(cmd)
        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT = None
        SoVITS_dropdown_update, GPT_dropdown_update = change_choices()
        yield (
            process_info(process_name_gpt, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            SoVITS_dropdown_update,
            GPT_dropdown_update,
        )
    else:
        yield (
            process_info(process_name_gpt, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
        )


def close1Bb():
    global p_train_GPT
    if p_train_GPT is not None:
        kill_process(p_train_GPT.pid, process_name_gpt)
        p_train_GPT = None
    return (
        process_info(process_name_gpt, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps_slice = []
process_name_slice = i18n("语音切分")


def open_slice(inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha, n_parts):
    global ps_slice
    inp = my_utils.clean_path(inp)
    opt_root = my_utils.clean_path(opt_root)
    check_for_existance([inp])
    if os.path.exists(inp) == False:
        yield (
            i18n("输入路径不存在"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return
    if os.path.isfile(inp):
        n_parts = 1
    elif os.path.isdir(inp):
        pass
    else:
        yield (
            i18n("输入路径存在但不可用"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return
    if ps_slice == []:
        for i_part in range(n_parts):
            cmd = '"%s" -s tools/slice_audio.py "%s" "%s" %s %s %s %s %s %s %s %s %s' % (
                python_exec,
                inp,
                opt_root,
                threshold,
                min_length,
                min_interval,
                hop_size,
                max_sil_kept,
                _max,
                alpha,
                i_part,
                n_parts,
            )
            print(cmd)
            p = Popen(cmd, shell=True)
            ps_slice.append(p)
        yield (
            process_info(process_name_slice, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        for p in ps_slice:
            p.wait()
        ps_slice = []
        yield (
            process_info(process_name_slice, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update", "value": opt_root},
            {"__type__": "update", "value": opt_root},
            {"__type__": "update", "value": opt_root},
        )
    else:
        yield (
            process_info(process_name_slice, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )


def close_slice():
    global ps_slice
    if ps_slice != []:
        for p_slice in ps_slice:
            try:
                kill_process(p_slice.pid, process_name_slice)
            except:
                traceback.print_exc()
        ps_slice = []
    return (
        process_info(process_name_slice, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps1a = []
process_name_1a = i18n("文本分词与特征提取")


def open1a(inp_text, inp_wav_dir, exp_name, gpu_numbers, bert_pretrained_dir):
    global ps1a
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    if ps1a == []:
        opt_dir = "%s/%s" % (exp_root, exp_name)
        config = {
            "inp_text": inp_text,
            "inp_wav_dir": inp_wav_dir,
            "exp_name": exp_name,
            "opt_dir": opt_dir,
            "bert_pretrained_dir": bert_pretrained_dir,
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                    "is_half": str(is_half),
                }
            )
            os.environ.update(config)
            cmd = '"%s" -s GPT_SoVITS/prepare_datasets/1-get-text.py' % python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1a.append(p)
        yield (
            process_info(process_name_1a, "running"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        for p in ps1a:
            p.wait()
        opt = []
        for i_part in range(all_parts):
            txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
            with open(txt_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(txt_path)
        path_text = "%s/2-name2text.txt" % opt_dir
        with open(path_text, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1a = []
        if len("".join(opt)) > 0:
            yield (
                process_info(process_name_1a, "finish"),
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
        else:
            yield (
                process_info(process_name_1a, "failed"),
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
    else:
        yield (
            process_info(process_name_1a, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )


def close1a():
    global ps1a
    if ps1a != []:
        for p1a in ps1a:
            try:
                kill_process(p1a.pid, process_name_1a)
            except:
                traceback.print_exc()
        ps1a = []
    return (
        process_info(process_name_1a, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


sv_path = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
ps1b = []
process_name_1b = i18n("语音自监督特征提取")


def open1b(version, inp_text, inp_wav_dir, exp_name, gpu_numbers, ssl_pretrained_dir):
    global ps1b
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    if ps1b == []:
        config = {
            "inp_text": inp_text,
            "inp_wav_dir": inp_wav_dir,
            "exp_name": exp_name,
            "opt_dir": "%s/%s" % (exp_root, exp_name),
            "cnhubert_base_dir": ssl_pretrained_dir,
            "sv_path": sv_path,
            "is_half": str(is_half),
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                }
            )
            os.environ.update(config)
            cmd = '"%s" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py' % python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1b.append(p)
        yield (
            process_info(process_name_1b, "running"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        for p in ps1b:
            p.wait()
        ps1b = []
        if "Pro" in version:
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                    }
                )
                os.environ.update(config)
                cmd = '"%s" -s GPT_SoVITS/prepare_datasets/2-get-sv.py' % python_exec
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1b.append(p)
            for p in ps1b:
                p.wait()
            ps1b = []
        yield (
            process_info(process_name_1b, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
    else:
        yield (
            process_info(process_name_1b, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )


def close1b():
    global ps1b
    if ps1b != []:
        for p1b in ps1b:
            try:
                kill_process(p1b.pid, process_name_1b)
            except:
                traceback.print_exc()
        ps1b = []
    return (
        process_info(process_name_1b, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps1c = []
process_name_1c = i18n("语义Token提取")


def open1c(version, inp_text, inp_wav_dir, exp_name, gpu_numbers, pretrained_s2G_path):
    global ps1c
    inp_text = my_utils.clean_path(inp_text)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    if ps1c == []:
        opt_dir = "%s/%s" % (exp_root, exp_name)
        config_file = (
            "GPT_SoVITS/configs/s2.json"
            if version not in {"v2Pro", "v2ProPlus"}
            else f"GPT_SoVITS/configs/s2{version}.json"
        )
        config = {
            "inp_text": inp_text,
            "exp_name": exp_name,
            "opt_dir": opt_dir,
            "pretrained_s2G": pretrained_s2G_path,
            "s2config_path": config_file,
            "is_half": str(is_half),
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                }
            )
            os.environ.update(config)
            cmd = '"%s" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py' % python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1c.append(p)
        yield (
            process_info(process_name_1c, "running"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        for p in ps1c:
            p.wait()
        opt = ["item_name\tsemantic_audio"]
        path_semantic = "%s/6-name2semantic.tsv" % opt_dir
        for i_part in range(all_parts):
            semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
            with open(semantic_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(semantic_path)
        with open(path_semantic, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1c = []
        yield (
            process_info(process_name_1c, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
    else:
        yield (
            process_info(process_name_1c, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )


def close1c():
    global ps1c
    if ps1c != []:
        for p1c in ps1c:
            try:
                kill_process(p1c.pid, process_name_1c)
            except:
                traceback.print_exc()
        ps1c = []
    return (
        process_info(process_name_1c, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps1abc = []
process_name_1abc = i18n("训练集格式化一键三连")


def open1abc(
    version,
    inp_text,
    inp_wav_dir,
    exp_name,
    gpu_numbers1a,
    gpu_numbers1Ba,
    gpu_numbers1c,
    bert_pretrained_dir,
    ssl_pretrained_dir,
    pretrained_s2G_path,
):
    global ps1abc
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    if ps1abc == []:
        opt_dir = "%s/%s" % (exp_root, exp_name)
        try:
            #############################1a
            path_text = "%s/2-name2text.txt" % opt_dir
            if os.path.exists(path_text) == False or (
                os.path.exists(path_text) == True
                and len(open(path_text, "r", encoding="utf8").read().strip("\n").split("\n")) < 2
            ):
                config = {
                    "inp_text": inp_text,
                    "inp_wav_dir": inp_wav_dir,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "bert_pretrained_dir": bert_pretrained_dir,
                    "is_half": str(is_half),
                }
                gpu_names = gpu_numbers1a.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" -s GPT_SoVITS/prepare_datasets/1-get-text.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield (
                    i18n("进度") + ": 1A-Doing",
                    {"__type__": "update", "visible": False},
                    {"__type__": "update", "visible": True},
                )
                for p in ps1abc:
                    p.wait()

                opt = []
                for i_part in range(all_parts):  # txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
                    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                    with open(txt_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                assert len("".join(opt)) > 0, process_info(process_name_1a, "failed")
            yield (
                i18n("进度") + ": 1A-Done",
                {"__type__": "update", "visible": False},
                {"__type__": "update", "visible": True},
            )
            ps1abc = []
            #############################1b
            config = {
                "inp_text": inp_text,
                "inp_wav_dir": inp_wav_dir,
                "exp_name": exp_name,
                "opt_dir": opt_dir,
                "cnhubert_base_dir": ssl_pretrained_dir,
                "sv_path": sv_path,
            }
            gpu_names = gpu_numbers1Ba.split("-")
            all_parts = len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                    }
                )
                os.environ.update(config)
                cmd = '"%s" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py' % python_exec
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1abc.append(p)
            yield (
                i18n("进度") + ": 1A-Done, 1B-Doing",
                {"__type__": "update", "visible": False},
                {"__type__": "update", "visible": True},
            )
            for p in ps1abc:
                p.wait()
            ps1abc = []
            if "Pro" in version:
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" -s GPT_SoVITS/prepare_datasets/2-get-sv.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                for p in ps1abc:
                    p.wait()
                ps1abc = []
            yield (
                i18n("进度") + ": 1A-Done, 1B-Done",
                {"__type__": "update", "visible": False},
                {"__type__": "update", "visible": True},
            )
            #############################1c
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if os.path.exists(path_semantic) == False or (
                os.path.exists(path_semantic) == True and os.path.getsize(path_semantic) < 31
            ):
                config_file = (
                    "GPT_SoVITS/configs/s2.json"
                    if version not in {"v2Pro", "v2ProPlus"}
                    else f"GPT_SoVITS/configs/s2{version}.json"
                )
                config = {
                    "inp_text": inp_text,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "pretrained_s2G": pretrained_s2G_path,
                    "s2config_path": config_file,
                }
                gpu_names = gpu_numbers1c.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield (
                    i18n("进度") + ": 1A-Done, 1B-Done, 1C-Doing",
                    {"__type__": "update", "visible": False},
                    {"__type__": "update", "visible": True},
                )
                for p in ps1abc:
                    p.wait()

                opt = ["item_name\tsemantic_audio"]
                for i_part in range(all_parts):
                    semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
                    with open(semantic_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)
                with open(path_semantic, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                yield (
                    i18n("进度") + ": 1A-Done, 1B-Done, 1C-Done",
                    {"__type__": "update", "visible": False},
                    {"__type__": "update", "visible": True},
                )
            ps1abc = []
            yield (
                process_info(process_name_1abc, "finish"),
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
        except:
            traceback.print_exc()
            close1abc()
            yield (
                process_info(process_name_1abc, "failed"),
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
    else:
        yield (
            process_info(process_name_1abc, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )


def close1abc():
    global ps1abc
    if ps1abc != []:
        for p1abc in ps1abc:
            try:
                kill_process(p1abc.pid, process_name_1abc)
            except:
                traceback.print_exc()
        ps1abc = []
    return (
        process_info(process_name_1abc, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


def switch_version(version_):
    os.environ["version"] = version_
    global version
    version = version_
    if pretrained_sovits_name[version] != "" and pretrained_gpt_name[version] != "":
        ...
    else:
        gr.Warning(i18n("未下载模型") + ": " + version.upper())
    set_default()
    return (
        {"__type__": "update", "value": pretrained_sovits_name[version]},
        {"__type__": "update", "value": pretrained_sovits_name[version].replace("s2G", "s2D") if version not in v3v4set else "", "visible": version not in v3v4set},
        {"__type__": "update", "value": pretrained_gpt_name[version]},
        {"__type__": "update", "value": pretrained_gpt_name[version]},
        {"__type__": "update", "value": pretrained_sovits_name[version]},
        {"__type__": "update", "value": default_batch_size, "maximum": default_max_batch_size},
        {"__type__": "update", "value": default_sovits_epoch, "maximum": max_sovits_epoch},
        {"__type__": "update", "value": default_sovits_save_every_epoch, "maximum": max_sovits_save_every_epoch},
        {"__type__": "update", "visible": True if version not in v3v4set else False},
        {
            "__type__": "update",
            "value": False if not if_force_ckpt else True,
            "interactive": True if not if_force_ckpt else False,
        },
        {"__type__": "update", "interactive": True, "value": False},
        {"__type__": "update", "visible": True if version in v3v4set else False},
    )  # {'__type__': 'update', "interactive": False if version in v3v4set else True, "value": False}, \ ####batch infer


if os.path.exists("GPT_SoVITS/text/G2PWModel"):
    ...
else:
    cmd = '"%s" -s GPT_SoVITS/download.py' % python_exec
    p = Popen(cmd, shell=True)
    p.wait()


def sync(text):
    return {"__type__": "update", "value": text}


with gr.Blocks(title="GPT-SoVITS WebUI", analytics_enabled=False, js=js, css=css) as app:
    gr.HTML(
        top_html.format(
            i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责.")
            + i18n("如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录LICENSE.")
        ),
        elem_classes="markdown",
    )

    with gr.Tabs(selected="tts"):
        with gr.TabItem("0-" + i18n("前置数据集获取工具"), id="tools"):  # 提前随机切片防止uvr5爆内存->uvr5->slicer->asr->打标
            with gr.Accordion(label="0a-" + i18n("UVR5人声伴奏分离&去混响去延迟工具")):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            uvr5_info = gr.Textbox(label=process_info(process_name_uvr5, "info"))
                    open_uvr5 = gr.Button(
                        value=process_info(process_name_uvr5, "open"), variant="primary", visible=True
                    )
                    close_uvr5 = gr.Button(
                        value=process_info(process_name_uvr5, "close"), variant="primary", visible=False
                    )

            with gr.Accordion(label="0b-" + i18n("语音切分工具")):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            slice_inp_path = gr.Textbox(label=i18n("音频自动切分输入路径，可文件可文件夹"), value="")
                            slice_opt_root = gr.Textbox(
                                label=i18n("切分后的子音频的输出根目录"), value="output/slicer_opt"
                            )
                        with gr.Row():
                            threshold = gr.Textbox(
                                label=i18n("threshold:音量小于这个值视作静音的备选切割点"), value="-34"
                            )
                            min_length = gr.Textbox(
                                label=i18n("min_length:每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值"),
                                value="4000",
                            )
                            min_interval = gr.Textbox(label=i18n("min_interval:最短切割间隔"), value="300")
                            hop_size = gr.Textbox(
                                label=i18n("hop_size:怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）"),
                                value="10",
                            )
                            max_sil_kept = gr.Textbox(label=i18n("max_sil_kept:切完后静音最多留多长"), value="500")
                        with gr.Row():
                            _max = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                label=i18n("max:归一化后最大值多少"),
                                value=0.9,
                                interactive=True,
                            )
                            alpha = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                label=i18n("alpha_mix:混多少比例归一化后音频进来"),
                                value=0.25,
                                interactive=True,
                            )
                        with gr.Row():
                            n_process = gr.Slider(
                                minimum=1,
                                maximum=n_cpu,
                                step=1,
                                label=i18n("切割使用的进程数"),
                                value=4,
                                interactive=True,
                            )
                            slicer_info = gr.Textbox(label=process_info(process_name_slice, "info"))
                    open_slicer_button = gr.Button(
                        value=process_info(process_name_slice, "open"), variant="primary", visible=True
                    )
                    close_slicer_button = gr.Button(
                        value=process_info(process_name_slice, "close"), variant="primary", visible=False
                    )

            # gr.Markdown(value="0bb-" + i18n("语音降噪工具")+i18n("(不稳定，先别用，可能劣化模型效果！)"))
            with gr.Row(visible=False):
                with gr.Column(scale=3):
                    with gr.Row():
                        denoise_input_dir = gr.Textbox(label=i18n("输入文件夹路径"), value="")
                        denoise_output_dir = gr.Textbox(label=i18n("输出文件夹路径"), value="output/denoise_opt")
                    with gr.Row():
                        denoise_info = gr.Textbox(label=process_info(process_name_denoise, "info"))
                open_denoise_button = gr.Button(
                    value=process_info(process_name_denoise, "open"), variant="primary", visible=True
                )
                close_denoise_button = gr.Button(
                    value=process_info(process_name_denoise, "close"), variant="primary", visible=False
                )

            with gr.Accordion(label="0c-" + i18n("语音识别工具")):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            asr_inp_dir = gr.Textbox(
                                label=i18n("输入文件夹路径"),
                                value="",
                                placeholder="/content/drive/MyDrive/data/wav",
                                interactive=True,
                            )
                            asr_opt_dir = gr.Textbox(
                                label=i18n("输出文件夹路径"), value="output/asr_opt", interactive=True
                            )
                        with gr.Row():
                            asr_model = gr.Dropdown(
                                label=i18n("ASR 模型"),
                                choices=ASR_UI_CHOICES,
                                interactive=True,
                                value=DEFAULT_ASR_MODEL,
                            )
                            asr_size = gr.Dropdown(
                                label=i18n("ASR 模型尺寸"),
                                choices=asr_dict[DEFAULT_ASR_MODEL]["size"],
                                interactive=True,
                                value=DEFAULT_ASR_SIZE,
                            )
                            asr_lang = gr.Dropdown(
                                label=i18n("ASR 语言设置"),
                                choices=asr_dict[DEFAULT_ASR_MODEL]["lang"],
                                interactive=True,
                                value=DEFAULT_ASR_LANGUAGE,
                            )
                            asr_precision = gr.Dropdown(
                                label=i18n("数据类型精度"),
                                choices=asr_dict[DEFAULT_ASR_MODEL]["precision"],
                                interactive=True,
                                value=DEFAULT_ASR_PRECISION,
                            )
                        with gr.Row():
                            asr_info = gr.Textbox(label=process_info(process_name_asr, "info"))
                    open_asr_button = gr.Button(
                        value=process_info(process_name_asr, "open"), variant="primary", visible=True
                    )
                    close_asr_button = gr.Button(
                        value=process_info(process_name_asr, "close"), variant="primary", visible=False
                    )

                def change_lang_choices(key):  # 根据选择的模型修改可选的语言
                    return {"__type__": "update", "choices": asr_dict[key]["lang"], "value": asr_dict[key]["lang"][0]}

                def change_size_choices(key):  # 根据选择的模型修改可选的模型尺寸
                    return {"__type__": "update", "choices": asr_dict[key]["size"], "value": asr_dict[key]["size"][-1]}

                def change_precision_choices(key):  # 根据选择的模型修改可选的语言
                    if key == "Faster Whisper (多语种)":
                        if default_batch_size <= 4:
                            precision = "int8"
                        elif is_half:
                            precision = "float16"
                        else:
                            precision = "float32"
                    else:
                        precision = "float32"
                    return {"__type__": "update", "choices": asr_dict[key]["precision"], "value": precision}

                asr_model.change(change_lang_choices, [asr_model], [asr_lang])
                asr_model.change(change_size_choices, [asr_model], [asr_size])
                asr_model.change(change_precision_choices, [asr_model], [asr_precision])

            with gr.Accordion(label="0d-" + i18n("语音文本校对标注工具")):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            path_list = gr.Textbox(
                                label=i18n("标注文件路径 (含文件后缀 *.list)"),
                                value="D:\\RVC1006\\GPT-SoVITS\\raw\\xxx.list",
                                interactive=True,
                            )
                            label_info = gr.Textbox(label=process_info(process_name_subfix, "info"))
                    open_label = gr.Button(
                        value=process_info(process_name_subfix, "open"), variant="primary", visible=True
                    )
                    close_label = gr.Button(
                        value=process_info(process_name_subfix, "close"), variant="primary", visible=False
                    )

                open_label.click(change_label, [path_list], [label_info, open_label, close_label])
                close_label.click(change_label, [path_list], [label_info, open_label, close_label])
                open_uvr5.click(change_uvr5, [], [uvr5_info, open_uvr5, close_uvr5])
                close_uvr5.click(change_uvr5, [], [uvr5_info, open_uvr5, close_uvr5])

        with gr.TabItem(i18n("1-GPT-SoVITS-TTS"), id="tts"):
            with gr.Accordion(i18n("微调模型信息")):
                with gr.Row():
                    with gr.Row(equal_height=True):
                        exp_name = gr.Textbox(
                            label=i18n("*实验/模型名"),
                            value="xxx",
                            interactive=True,
                            scale=3,
                        )
                        gpu_info_box = gr.Textbox(
                            label=i18n("显卡信息"),
                            value=gpu_info,
                            visible=True,
                            interactive=False,
                            scale=5,
                        )
                        version_checkbox = gr.Radio(
                            label=i18n("训练模型的版本"),
                            value=version,
                            choices=["v2", "v2Pro", "v2ProPlus", "v4", "v5dev"],
                            scale=5,
                        )
            with gr.Accordion(label=i18n("预训练模型路径"), open=False):
                with gr.Row():
                    with gr.Row(equal_height=True):
                        pretrained_s1 = gr.Textbox(
                            label=i18n("预训练GPT模型路径"),
                            value=pretrained_gpt_name[version],
                            interactive=True,
                            lines=1,
                            max_lines=1,
                            scale=3,
                        )
                        pretrained_s2G = gr.Textbox(
                            label=i18n("预训练SoVITS-G模型路径"),
                            value=pretrained_sovits_name[version],
                            interactive=True,
                            lines=1,
                            max_lines=1,
                            scale=5,
                        )
                        pretrained_s2D = gr.Textbox(
                            label=i18n("预训练SoVITS-D模型路径"),
                            value=pretrained_sovits_name[version].replace("s2G", "s2D") if version not in v3v4set else "",
                            visible=version not in v3v4set,
                            interactive=True,
                            lines=1,
                            max_lines=1,
                            scale=5,
                        )

            with gr.Tabs(selected="finetune"):
                with gr.TabItem("1A-" + i18n("训练集格式化工具"), id="dataset"):
                    with gr.Accordion(
                        label="1A0-음성 전사 (.list 자동 생성, 선택)",
                        open=False,
                    ):
                        gr.Markdown(
                            """
이미 GPT-SoVITS 형식의 `.list` 파일이 있으면 이 단계는 건너뛰면 됩니다.

음성 파일만 있는 경우 폴더를 지정하고 ASR로 전사하면 아래의
**텍스트 표기 파일**과 **학습용 음성 폴더**가 자동으로 채워집니다.
다국어 사용을 고려해 Faster Whisper를 기본값으로 둡니다.
긴 원본 녹음은 필요하면 0-데이터셋 준비 도구의 음성 자르기를 먼저 사용하세요.
"""
                        )

                        with gr.Row():
                            prep_asr_inp_dir = gr.Textbox(
                                label="전사할 음성 폴더",
                                placeholder="/content/drive/MyDrive/data/wav",
                                interactive=True,
                            )
                            prep_asr_opt_dir = gr.Textbox(
                                label="전사 결과 저장 폴더",
                                value="output/asr_opt",
                                interactive=True,
                            )

                        with gr.Row():
                            prep_asr_model = gr.Dropdown(
                                label="ASR 엔진",
                                choices=ASR_UI_CHOICES,
                                value=DEFAULT_ASR_MODEL,
                                interactive=True,
                            )
                            prep_asr_size = gr.Dropdown(
                                label="모델 크기",
                                choices=asr_dict[DEFAULT_ASR_MODEL]["size"],
                                value=DEFAULT_ASR_SIZE,
                                interactive=True,
                            )
                            prep_asr_lang = gr.Dropdown(
                                label="언어",
                                choices=asr_dict[DEFAULT_ASR_MODEL]["lang"],
                                value=DEFAULT_ASR_LANGUAGE,
                                interactive=True,
                            )
                            prep_asr_precision = gr.Dropdown(
                                label="정밀도",
                                choices=asr_dict[DEFAULT_ASR_MODEL]["precision"],
                                value=DEFAULT_ASR_PRECISION,
                                interactive=True,
                            )

                        with gr.Row():
                            prep_asr_open = gr.Button(
                                value="음성 전사 시작",
                                variant="primary",
                                visible=True,
                            )
                            prep_asr_close = gr.Button(
                                value="음성 전사 중지",
                                variant="stop",
                                visible=False,
                            )

                        prep_asr_info = gr.Textbox(
                            label="전사 상태",
                            interactive=False,
                        )

                        prep_asr_model.change(
                            change_lang_choices,
                            [prep_asr_model],
                            [prep_asr_lang],
                        )
                        prep_asr_model.change(
                            change_size_choices,
                            [prep_asr_model],
                            [prep_asr_size],
                        )
                        prep_asr_model.change(
                            change_precision_choices,
                            [prep_asr_model],
                            [prep_asr_precision],
                        )

                    with gr.Accordion(label=i18n("输出logs/实验名目录下应有23456开头的文件和文件夹")):
                        with gr.Row():
                            with gr.Row():
                                inp_text = gr.Textbox(
                                    label=i18n("*文本标注文件"),
                                    value="",
                                    placeholder="/content/drive/MyDrive/data/train.list",
                                    interactive=True,
                                    scale=10,
                                )
                            with gr.Row():
                                inp_wav_dir = gr.Textbox(
                                    label=i18n("*训练集音频文件目录"),
                                    # value=r"D:\RVC1006\GPT-SoVITS\raw\xxx",
                                    interactive=True,
                                    placeholder=i18n(
                                        "填切割后音频所在目录！读取的音频文件完整路径=该目录-拼接-list文件里波形对应的文件名（不是全路径）。如果留空则使用.list文件里的绝对全路径。"
                                    ),
                                    scale=10,
                                )
    
                    with gr.Accordion(label="1Aa-" + process_name_1a):
                        with gr.Row():
                            with gr.Row():
                                gpu_numbers1a = gr.Textbox(
                                    label=i18n("GPU卡号以-分割，每个卡号一个进程"),
                                    value="%s-%s" % (gpus, gpus),
                                    interactive=True,
                                )
                            with gr.Row():
                                bert_pretrained_dir = gr.Textbox(
                                    label=i18n("预训练中文BERT模型路径"),
                                    value="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
                                    interactive=False,
                                    lines=2,
                                )
                            with gr.Row():
                                button1a_open = gr.Button(
                                    value=process_info(process_name_1a, "open"), variant="primary", visible=True
                                )
                                button1a_close = gr.Button(
                                    value=process_info(process_name_1a, "close"), variant="primary", visible=False
                                )
                            with gr.Row():
                                info1a = gr.Textbox(label=process_info(process_name_1a, "info"))
    
                    with gr.Accordion(label="1Ab-" + process_name_1b):
                        with gr.Row():
                            with gr.Row():
                                gpu_numbers1Ba = gr.Textbox(
                                    label=i18n("GPU卡号以-分割，每个卡号一个进程"),
                                    value="%s-%s" % (gpus, gpus),
                                    interactive=True,
                                )
                            with gr.Row():
                                cnhubert_base_dir = gr.Textbox(
                                    label=i18n("预训练SSL模型路径"),
                                    value="GPT_SoVITS/pretrained_models/chinese-hubert-base",
                                    interactive=False,
                                    lines=2,
                                )
                            with gr.Row():
                                button1b_open = gr.Button(
                                    value=process_info(process_name_1b, "open"), variant="primary", visible=True
                                )
                                button1b_close = gr.Button(
                                    value=process_info(process_name_1b, "close"), variant="primary", visible=False
                                )
                            with gr.Row():
                                info1b = gr.Textbox(label=process_info(process_name_1b, "info"))
    
                    with gr.Accordion(label="1Ac-" + process_name_1c):
                        with gr.Row():
                            with gr.Row():
                                gpu_numbers1c = gr.Textbox(
                                    label=i18n("GPU卡号以-分割，每个卡号一个进程"),
                                    value="%s-%s" % (gpus, gpus),
                                    interactive=True,
                                )
                            with gr.Row():
                                pretrained_s2G_ = gr.Textbox(
                                    label=i18n("预训练SoVITS-G模型路径"),
                                    value=pretrained_sovits_name[version],
                                    interactive=False,
                                    lines=2,
                                )
                            with gr.Row():
                                button1c_open = gr.Button(
                                    value=process_info(process_name_1c, "open"), variant="primary", visible=True
                                )
                                button1c_close = gr.Button(
                                    value=process_info(process_name_1c, "close"), variant="primary", visible=False
                                )
                            with gr.Row():
                                info1c = gr.Textbox(label=process_info(process_name_1c, "info"))
    
                    with gr.Accordion(label="1Aabc-" + process_name_1abc):
                        with gr.Row():
                            with gr.Row():
                                button1abc_open = gr.Button(
                                    value=process_info(process_name_1abc, "open"), variant="primary", visible=True
                                )
                                button1abc_close = gr.Button(
                                    value=process_info(process_name_1abc, "close"), variant="primary", visible=False
                                )
                            with gr.Row():
                                info1abc = gr.Textbox(label=process_info(process_name_1abc, "info"))
    

                    with gr.Accordion(
                        label="1Ad-감정 학습 데이터 준비 (선택)",
                        open=False,
                    ):
                        gr.Markdown(
                            """
일반 GPT-SoVITS 학습에는 필요하지 않습니다.

감정 제어를 학습하려면 위의 **텍스트 표기(.list)**와
**학습용 음성 폴더**를 그대로 사용합니다. 위의 Whisper/Fun-ASR은 **전사(ASR)** 용이고,
여기의 emotion2vec+는 **음성 감정 분류** 용이라 서로 별개의 단계입니다.

자동 생성은 같은 화자·같은 언어·**동일한 전사문**을 가진 음성을 묶고,
감정 분류 결과가 충분히 확실한 경우에만 서로 다른 감정의 pair를 만듭니다.
자동 결과가 마음에 들지 않으면 sidecar JSONL을 직접 수정하거나 수동 파일로
바꿔도 됩니다.
"""
                        )

                        with gr.Row():
                            emotion_auto_model = gr.Dropdown(
                                label="감정 분석 모델 (ASR과 별개)",
                                choices=[
                                    (
                                        "emotion2vec+ base (기본)",
                                        "iic/emotion2vec_plus_base",
                                    ),
                                    (
                                        "emotion2vec+ large (정확도 우선)",
                                        "iic/emotion2vec_plus_large",
                                    ),
                                ],
                                value="iic/emotion2vec_plus_base",
                                interactive=True,
                            )
                            emotion_auto_confidence = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                step=0.05,
                                label="자동 감정 라벨 최소 신뢰도",
                                value=0.45,
                                interactive=True,
                            )
                            emotion_auto_device = gr.Dropdown(
                                label="자동 감정 분석 장치",
                                choices=["auto", "cuda:0", "cpu"],
                                value="auto",
                                interactive=True,
                            )

                        with gr.Row():
                            emotion_sidecar = gr.Textbox(
                                label="감정 sidecar JSONL 경로 (자동 생성/수동 지정)",
                                value="",
                                placeholder=(
                                    "비워두면 .list 파일 옆에 "
                                    "*.emotion.auto.jsonl로 자동 생성"
                                ),
                                interactive=True,
                            )
                            emotion_dataset_id = gr.Textbox(
                                label="감정 데이터셋 ID",
                                value="emotion_dataset_01",
                                interactive=True,
                            )

                        with gr.Row():
                            emotion_auto_run = gr.Button(
                                value="음성에서 감정 pair 자동 생성",
                                variant="primary",
                            )
                            emotion_auto_stop = gr.Button(
                                value="자동 감정 분석 중지",
                                variant="stop",
                            )

                        emotion_auto_status = gr.Markdown(
                            value=(
                                "동일 전사문 그룹 탐색 → emotion2vec+ 감정 분석 → "
                                "신뢰도 필터 → reference 예약 → sidecar 생성"
                            )
                        )

                        with gr.Row():
                            emotion_dataset_dir = gr.Textbox(
                                label="감정 canonical dataset 출력 폴더",
                                value="",
                                placeholder=(
                                    "비워두면 /content/"
                                    "<데이터셋 ID>_emotion_dataset"
                                ),
                                interactive=True,
                            )
                            emotion_reuse_dataset = gr.Checkbox(
                                label="기존 감정 canonical dataset 재사용",
                                value=False,
                                interactive=True,
                            )

                        emotion_auto_run.click(
                            run_auto_emotion_sidecar,
                            [
                                inp_text,
                                inp_wav_dir,
                                emotion_sidecar,
                                emotion_auto_model,
                                emotion_auto_confidence,
                                emotion_auto_device,
                            ],
                            [
                                emotion_auto_status,
                                emotion_sidecar,
                            ],
                        )

                        emotion_auto_stop.click(
                            stop_emotion_training,
                            [],
                            [emotion_auto_status],
                            queue=False,
                        )

                prep_asr_open.click(
                    open_asr,
                    [
                        prep_asr_inp_dir,
                        prep_asr_opt_dir,
                        prep_asr_model,
                        prep_asr_size,
                        prep_asr_lang,
                        prep_asr_precision,
                    ],
                    [
                        prep_asr_info,
                        prep_asr_open,
                        prep_asr_close,
                        path_list,
                        inp_text,
                        inp_wav_dir,
                    ],
                )
                prep_asr_close.click(
                    close_asr,
                    [],
                    [
                        prep_asr_info,
                        prep_asr_open,
                        prep_asr_close,
                    ],
                )

                pretrained_s2G.change(sync, [pretrained_s2G], [pretrained_s2G_])
                open_asr_button.click(
                    open_asr,
                    [asr_inp_dir, asr_opt_dir, asr_model, asr_size, asr_lang, asr_precision],
                    [asr_info, open_asr_button, close_asr_button, path_list, inp_text, inp_wav_dir],
                )
                close_asr_button.click(close_asr, [], [asr_info, open_asr_button, close_asr_button])
                open_slicer_button.click(
                    open_slice,
                    [
                        slice_inp_path,
                        slice_opt_root,
                        threshold,
                        min_length,
                        min_interval,
                        hop_size,
                        max_sil_kept,
                        _max,
                        alpha,
                        n_process,
                    ],
                    [slicer_info, open_slicer_button, close_slicer_button, asr_inp_dir, denoise_input_dir, inp_wav_dir],
                )
                close_slicer_button.click(close_slice, [], [slicer_info, open_slicer_button, close_slicer_button])
                open_denoise_button.click(
                    open_denoise,
                    [denoise_input_dir, denoise_output_dir],
                    [denoise_info, open_denoise_button, close_denoise_button, asr_inp_dir, inp_wav_dir],
                )
                close_denoise_button.click(close_denoise, [], [denoise_info, open_denoise_button, close_denoise_button])
    
                button1a_open.click(
                    open1a,
                    [inp_text, inp_wav_dir, exp_name, gpu_numbers1a, bert_pretrained_dir],
                    [info1a, button1a_open, button1a_close],
                )
                button1a_close.click(close1a, [], [info1a, button1a_open, button1a_close])
                button1b_open.click(
                    open1b,
                    [version_checkbox, inp_text, inp_wav_dir, exp_name, gpu_numbers1Ba, cnhubert_base_dir],
                    [info1b, button1b_open, button1b_close],
                )
                button1b_close.click(close1b, [], [info1b, button1b_open, button1b_close])
                button1c_open.click(
                    open1c,
                    [version_checkbox, inp_text, inp_wav_dir, exp_name, gpu_numbers1c, pretrained_s2G],
                    [info1c, button1c_open, button1c_close],
                )
                button1c_close.click(close1c, [], [info1c, button1c_open, button1c_close])
                button1abc_open.click(
                    open1abc,
                    [
                        version_checkbox,
                        inp_text,
                        inp_wav_dir,
                        exp_name,
                        gpu_numbers1a,
                        gpu_numbers1Ba,
                        gpu_numbers1c,
                        bert_pretrained_dir,
                        cnhubert_base_dir,
                        pretrained_s2G,
                    ],
                    [info1abc, button1abc_open, button1abc_close],
                )
                button1abc_close.click(close1abc, [], [info1abc, button1abc_open, button1abc_close])
    
                with gr.TabItem("1B-" + i18n("微调训练"), id="finetune"):
                    with gr.Accordion(label="1Ba-" + i18n("SoVITS 训练: 模型权重文件在 SoVITS_weights/")):
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    batch_size = gr.Slider(
                                        minimum=1,
                                        maximum=default_max_batch_size,
                                        step=1,
                                        label=i18n("每张显卡的batch_size"),
                                        value=default_batch_size,
                                        interactive=True,
                                    )
                                    total_epoch = gr.Slider(
                                        minimum=1,
                                        maximum=max_sovits_epoch,
                                        step=1,
                                        label=i18n("总训练轮数total_epoch，不建议太高"),
                                        value=default_sovits_epoch,
                                        interactive=True,
                                    )
                                with gr.Row():
                                    text_low_lr_rate = gr.Slider(
                                        minimum=0.2,
                                        maximum=0.6,
                                        step=0.05,
                                        label=i18n("文本模块学习率权重"),
                                        value=0.4,
                                        visible=True if version not in v3v4set else False,
                                    )  # v3v4 not need
                                    lora_rank = gr.Radio(
                                        label=i18n("LoRA秩"),
                                        value="32",
                                        choices=["16", "32", "64", "128"],
                                        visible=True if version in v3v4set else False,
                                    )  # v1v2 not need
                                    save_every_epoch = gr.Slider(
                                        minimum=1,
                                        maximum=max_sovits_save_every_epoch,
                                        step=1,
                                        label=i18n("保存频率save_every_epoch"),
                                        value=default_sovits_save_every_epoch,
                                        interactive=True,
                                    )
                            with gr.Column():
                                with gr.Column():
                                    if_save_latest = gr.Checkbox(
                                        label=i18n("是否仅保存最新的权重文件以节省硬盘空间"),
                                        value=True,
                                        interactive=True,
                                        show_label=True,
                                    )
                                    if_save_every_weights = gr.Checkbox(
                                        label=i18n("是否在每次保存时间点将最终小模型保存至weights文件夹"),
                                        value=True,
                                        interactive=True,
                                        show_label=True,
                                    )
                                    if_grad_ckpt = gr.Checkbox(
                                        label="v3是否开启梯度检查点节省显存占用",
                                        value=False,
                                        interactive=True if version in v3v4set else False,
                                        show_label=True,
                                        visible=False,
                                    )  # 只有V3s2可以用
                                with gr.Row():
                                    gpu_numbers1Ba = gr.Textbox(
                                        label=i18n("GPU卡号以-分割，每个卡号一个进程"),
                                        value="%s" % (gpus),
                                        interactive=True,
                                    )
                        with gr.Row():
                            with gr.Row():
                                button1Ba_open = gr.Button(
                                    value=process_info(process_name_sovits, "open"), variant="primary", visible=True
                                )
                                button1Ba_close = gr.Button(
                                    value=process_info(process_name_sovits, "close"), variant="primary", visible=False
                                )
                            with gr.Row():
                                info1Ba = gr.Textbox(label=process_info(process_name_sovits, "info"))
                    with gr.Accordion(label="1Bb-" + i18n("GPT 训练: 模型权重文件在 GPT_weights/")):
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    batch_size1Bb = gr.Slider(
                                        minimum=1,
                                        maximum=40,
                                        step=1,
                                        label=i18n("每张显卡的batch_size"),
                                        value=default_batch_size_s1,
                                        interactive=True,
                                    )
                                    total_epoch1Bb = gr.Slider(
                                        minimum=2,
                                        maximum=50,
                                        step=1,
                                        label=i18n("总训练轮数total_epoch"),
                                        value=15,
                                        interactive=True,
                                    )
                                with gr.Row():
                                    save_every_epoch1Bb = gr.Slider(
                                        minimum=1,
                                        maximum=50,
                                        step=1,
                                        label=i18n("保存频率save_every_epoch"),
                                        value=5,
                                        interactive=True,
                                    )
                                    if_dpo = gr.Checkbox(
                                        label=i18n("是否开启DPO训练选项(实验性)"),
                                        value=False,
                                        interactive=True,
                                        show_label=True,
                                    )
                            with gr.Column():
                                with gr.Column():
                                    if_save_latest1Bb = gr.Checkbox(
                                        label=i18n("是否仅保存最新的权重文件以节省硬盘空间"),
                                        value=True,
                                        interactive=True,
                                        show_label=True,
                                    )
                                    if_save_every_weights1Bb = gr.Checkbox(
                                        label=i18n("是否在每次保存时间点将最终小模型保存至weights文件夹"),
                                        value=True,
                                        interactive=True,
                                        show_label=True,
                                    )
                                with gr.Row():
                                    gpu_numbers1Bb = gr.Textbox(
                                        label=i18n("GPU卡号以-分割，每个卡号一个进程"),
                                        value="%s" % (gpus),
                                        interactive=True,
                                    )
                        with gr.Row():
                            with gr.Row():
                                button1Bb_open = gr.Button(
                                    value=process_info(process_name_gpt, "open"), variant="primary", visible=True
                                )
                                button1Bb_close = gr.Button(
                                    value=process_info(process_name_gpt, "close"), variant="primary", visible=False
                                )
                            with gr.Row():
                                info1Bb = gr.Textbox(label=process_info(process_name_gpt, "info"))
    

                    with gr.Accordion(
                        label="1Bc-감정 제어 학습 (V5dev, 선택)",
                        open=False,
                    ):
                        gr.Markdown(
                            """
### V5dev Emotion Controller

기존 GPT-SoVITS 파인튜닝 흐름은 그대로 유지합니다.
감정 제어가 필요한 경우에만 1A에서 준비한 감정 pair 메타데이터를 사용해
작은 emotion controller를 추가로 학습합니다.

GPT/S1을 파인튜닝했다면 아래 체크포인트에는 **최종 GPT/S1 모델**을 선택해야 합니다.
"""
                        )

                        with gr.Row():
                            emotion_feature_root = gr.Textbox(
                                label="V5 전처리 feature 폴더",
                                value="",
                                placeholder=(
                                    "비워두면 /content/"
                                    "<데이터셋 ID>_emotion_features"
                                ),
                                interactive=True,
                            )
                            emotion_controller_output = gr.Textbox(
                                label="Controller 저장 경로",
                                value="",
                                placeholder=(
                                    "비워두면 Google Drive에 "
                                    "<데이터셋 ID>_emotion_controller.pt"
                                ),
                                interactive=True,
                            )

                        with gr.Row():
                            emotion_s1_checkpoint = gr.Textbox(
                                label=(
                                    "Controller에 사용할 GPT/S1 체크포인트 "
                                    "(추론 모델과 정확히 일치해야 함)"
                                ),
                                value="GPT_SoVITS/pretrained_models/s1v3.ckpt",
                                interactive=True,
                            )

                        with gr.Row():
                            emotion_names = gr.Textbox(
                                label="감정 목록 (선택)",
                                value="",
                                placeholder=(
                                    "비워두면 실제 pair 데이터에서 자동 감지 "
                                    "(예: neutral,happy,sad)"
                                ),
                                interactive=True,
                            )
                            emotion_steps = gr.Number(
                                label="학습 steps",
                                value=1000,
                                minimum=1,
                                precision=0,
                                interactive=True,
                            )
                            emotion_gpu = gr.Textbox(
                                label="GPU 번호",
                                value=str(default_gpu_numbers),
                                interactive=True,
                            )

                        gr.Markdown(
                            (
                                "**S2 semantic 기준:** "
                                "`GPT_SoVITS/pretrained_models/"
                                "gsv-v5-pretrained/s2Gv5dev.pth`  "
                                "(V5dev LoRA도 이 base token 공간을 사용)"
                            )
                        )

                        with gr.Row():
                            emotion_run = gr.Button(
                                value="감정 제어 학습 시작",
                                variant="primary",
                            )
                            emotion_stop = gr.Button(
                                value="감정 학습 중지",
                                variant="stop",
                            )

                        emotion_status = gr.Markdown(
                            value=(
                                "1A 감정 데이터 확인 → V5dev feature 전처리 → "
                                "reference prompt 생성 → controller 학습"
                            )
                        )

                    emotion_run.click(
                        run_emotion_training_pipeline,
                        [
                            inp_text,
                            inp_wav_dir,
                            emotion_sidecar,
                            emotion_dataset_id,
                            emotion_dataset_dir,
                            emotion_feature_root,
                            emotion_controller_output,
                            emotion_s1_checkpoint,
                            emotion_names,
                            emotion_steps,
                            emotion_gpu,
                            emotion_reuse_dataset,
                        ],
                        [emotion_status],
                    )

                    emotion_stop.click(
                        stop_emotion_training,
                        [],
                        [emotion_status],
                        queue=False,
                    )

                button1Ba_close.click(close1Ba, [], [info1Ba, button1Ba_open, button1Ba_close])
                button1Bb_close.click(close1Bb, [], [info1Bb, button1Bb_open, button1Bb_close])
    
                with gr.TabItem("1C-" + i18n("推理"), id="inference"):
                    gr.Markdown(
                        value=i18n(
                            "选择训练完存放在SoVITS_weights和GPT_weights下的模型。默认的几个是底模，体验5秒Zero Shot TTS不训练推理用。"
                        )
                    )
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Row():
                                GPT_dropdown = gr.Dropdown(
                                    label=i18n("GPT模型列表"),
                                    choices=GPT_names,
                                    value=default_gpt_choice,
                                    interactive=True,
                                )
                                SoVITS_dropdown = gr.Dropdown(
                                    label=i18n("SoVITS模型列表"),
                                    choices=SoVITS_names,
                                    value=default_sovits_choice,
                                    interactive=True,
                                )
                        with gr.Column(scale=2):
                            with gr.Row():
                                gpu_number_1C = gr.Textbox(
                                    label=i18n("GPU卡号,只能填1个整数"), value=gpus, interactive=True
                                )
                                refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary")
                        refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])
                    with gr.Row(equal_height=True):
                        with gr.Row():
                            batched_infer_enabled = gr.Checkbox(
                                label=i18n("启用并行推理版本"), value=False, interactive=True, show_label=True
                            )
                            open_tts = gr.Button(
                                value=process_info(process_name_tts, "open"), variant="primary", visible=True
                            )
                            close_tts = gr.Button(
                                value=process_info(process_name_tts, "close"), variant="primary", visible=False
                            )
                        with gr.Column():
                            tts_info = gr.Markdown(value=process_info(process_name_tts, "info"))
                        open_tts.click(
                            change_tts_inference,
                            [
                                bert_pretrained_dir,
                                cnhubert_base_dir,
                                gpu_number_1C,
                                GPT_dropdown,
                                SoVITS_dropdown,
                                batched_infer_enabled,
                            ],
                            [tts_info, open_tts, close_tts],
                        )
                        close_tts.click(
                            change_tts_inference,
                            [
                                bert_pretrained_dir,
                                cnhubert_base_dir,
                                gpu_number_1C,
                                GPT_dropdown,
                                SoVITS_dropdown,
                                batched_infer_enabled,
                            ],
                            [tts_info, open_tts, close_tts],
                        )
                button1Ba_open.click(
                    open1Ba,
                    [
                        version_checkbox,
                        batch_size,
                        total_epoch,
                        exp_name,
                        text_low_lr_rate,
                        if_save_latest,
                        if_save_every_weights,
                        save_every_epoch,
                        gpu_numbers1Ba,
                        pretrained_s2G,
                        pretrained_s2D,
                        if_grad_ckpt,
                        lora_rank,
                    ],
                    [info1Ba, button1Ba_open, button1Ba_close, SoVITS_dropdown, GPT_dropdown],
                )
                button1Bb_open.click(
                    open1Bb,
                    [
                        batch_size1Bb,
                        total_epoch1Bb,
                        exp_name,
                        if_dpo,
                        if_save_latest1Bb,
                        if_save_every_weights1Bb,
                        save_every_epoch1Bb,
                        gpu_numbers1Bb,
                        pretrained_s1,
                    ],
                    [info1Bb, button1Bb_open, button1Bb_close, SoVITS_dropdown, GPT_dropdown],
                )
                version_checkbox.change(
                    switch_version,
                    [version_checkbox],
                    [
                        pretrained_s2G,
                        pretrained_s2D,
                        pretrained_s1,
                        GPT_dropdown,
                        SoVITS_dropdown,
                        batch_size,
                        total_epoch,
                        save_every_epoch,
                        text_low_lr_rate,
                        if_grad_ckpt,
                        batched_infer_enabled,
                        lora_rank,
                    ],
                )
    
        with gr.TabItem(i18n("2-GPT-SoVITS-变声"), id="voice"):
            gr.Markdown(value=i18n("施工中，请静候佳音"))

    app.queue().launch(  # concurrency_count=511, max_size=1022
        server_name="0.0.0.0",
        inbrowser=True,
        share=is_share,
        server_port=webui_port_main,
        # quiet=True,
    )
