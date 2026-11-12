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

bundled_nltk_data = Path(now_dir) / "runtime" / "nltk_data"
if bundled_nltk_data.is_dir():
    existing_nltk_data = os.environ.get("NLTK_DATA", "").strip()
    nltk_paths = [str(bundled_nltk_data)]
    if existing_nltk_data:
        nltk_paths.append(existing_nltk_data)
    os.environ["NLTK_DATA"] = os.pathsep.join(nltk_paths)

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

from tools.assets import css, js, runtime_status_patch_js, top_html
from GPT_SoVITS.text.multilingual_bert import (
    ALIGNMENT_VERSION as MULTILINGUAL_BERT_ALIGNMENT_VERSION,
)
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
    latest_gpt_checkpoint_for_experiment,
    validate_emotion_dataset,
    write_v5_tts_config,
)
from tools.my_utils import check_details, check_for_existance
from tools.runtime_status import (
    build_runtime_status_html,
    build_runtime_status_payload,
)
from tools.preprocess_contract import (
    build_dataset_fingerprint,
    build_stage_contract,
    contract_matches,
    file_sha256,
    prepare_training_resume_state,
    quick_path_signature,
    stage_matches_dataset,
    read_stage_contract,
    stage_contract_path,
    validate_dataset_fingerprint,
    write_stage_contract,
)

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
denoise_cancel_requested = False
p_tts_inference = None
tts_inference_cancel_requested = False
p_emotion_training = None
emotion_task_cancel_requested = False
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
        cmd = [
            "taskkill",
            "/t",
            "/f",
            "/pid",
            str(pid),
        ]
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
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



class EmotionTaskCancelled(EmotionPipelineError):
    pass


def _clean_user_path(value):
    text = str(value or "").strip().replace(chr(0x202A), "")
    return text.strip(chr(39) + chr(34))


def _file_signature(path):
    path = Path(path)
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _nonempty_line_count(path):
    path = Path(path)
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def _text_feature_config(
    use_multilingual_bert,
    multilingual_bert_model,
):
    enabled = bool(use_multilingual_bert)
    return {
        "format_version": 1,
        "multilingual_bert": enabled,
        "multilingual_bert_alignment": (
            MULTILINGUAL_BERT_ALIGNMENT_VERSION
            if enabled
            else None
        ),
        "multilingual_bert_model": (
            str(multilingual_bert_model).strip()
            if enabled
            else None
        ),
    }


def _write_text_feature_config(
    path_text,
    use_multilingual_bert,
    multilingual_bert_model,
):
    marker = Path(path_text).parent / "text_feature_config.json"
    payload = _text_feature_config(
        use_multilingual_bert,
        multilingual_bert_model,
    )
    temp = marker.with_name(f".{marker.name}.tmp-{os.getpid()}")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, marker)


def _text_preprocess_cache_valid(
    path_text,
    inp_text,
    feature_config=None,
):
    output_count = _nonempty_line_count(path_text)
    input_count = _nonempty_line_count(inp_text)
    if not (input_count > 0 and output_count == input_count):
        return False
    if feature_config is None:
        return True

    marker = Path(path_text).parent / "text_feature_config.json"
    if not marker.is_file():
        return False
    try:
        current = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return current == feature_config


def _resolve_repo_path(value):
    text = _clean_user_path(value)
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path(now_dir) / path
    return path.resolve()


def _preprocess_contract(
    stage,
    inp_text,
    inp_wav_dir,
    version_name,
    settings=None,
):
    return build_stage_contract(
        stage=stage,
        list_path=_clean_user_path(inp_text),
        wav_root=_clean_user_path(inp_wav_dir) or None,
        version=stage_version_key(stage, version_name),
        settings=settings or {},
    )


def _clear_preprocess_stage_outputs(exp_dir, stage):
    exp_dir = Path(exp_dir)
    marker = stage_contract_path(exp_dir, stage)

    if stage == "1a":
        targets = [
            exp_dir / "2-name2text.txt",
            exp_dir / "3-bert",
            exp_dir / "text_feature_config.json",
        ]
        targets.extend(exp_dir.glob("2-name2text-*.txt"))
    elif stage == "1b":
        targets = [
            exp_dir / "4-cnhubert",
            exp_dir / "5-wav32k",
            exp_dir / "7-sv",
        ]
    elif stage == "1c":
        targets = list(exp_dir.glob("6-name2semantic*.tsv"))
    else:
        targets = []

    for target in targets:
        try:
            if Path(target).is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif Path(target).exists():
                Path(target).unlink()
        except OSError:
            traceback.print_exc()

    try:
        if marker.exists():
            marker.unlink()
    except OSError:
        traceback.print_exc()


def _stage_dataset_error(contract):
    errors = validate_dataset_fingerprint(
        contract.get("dataset", {})
    )
    if not errors:
        return None
    return "; ".join(errors)


def _ensure_stage_contract(exp_dir, expected):
    marker = stage_contract_path(exp_dir, expected["stage"])
    if contract_matches(marker, expected):
        return False
    _clear_preprocess_stage_outputs(exp_dir, expected["stage"])
    return True


def _validate_training_dataset_contract(
    exp_dir,
    inp_text,
    inp_wav_dir,
    version_name,
    required_stages,
):
    exp_dir = Path(exp_dir)
    try:
        current = _preprocess_contract(
            "_current",
            inp_text,
            inp_wav_dir,
            version_name,
        )
    except (OSError, ValueError, FileNotFoundError) as exc:
        return f"현재 학습 데이터셋 fingerprint 생성 실패: {exc}"

    source_error = _stage_dataset_error(current)
    if source_error:
        return "현재 .list/WAV 데이터셋이 유효하지 않습니다: " + source_error

    contracts = []
    for stage in required_stages:
        marker = stage_contract_path(exp_dir, stage)
        contract = read_stage_contract(marker)
        if contract is None:
            return (
                f"{stage} 전처리 provenance가 없습니다. "
                "현재 데이터로 전처리를 다시 실행하세요."
            )
        expected_stage_version = stage_version_key(
            stage,
            version_name,
        )
        if str(contract.get("version")) != str(expected_stage_version):
            return (
                f"{stage} 전처리 모드({contract.get('version')})와 "
                f"현재 요구 모드({expected_stage_version})가 다릅니다."
            )
        if not stage_matches_dataset(
            contract,
            current.get("dataset", {}),
        ):
            return (
                f"{stage} 전처리 입력과 현재 데이터셋의 해당 의존성이 "
                "다릅니다. 필요한 단계만 다시 전처리하세요."
            )

        if stage == "1c":
            upstream_1b = read_stage_contract(
                stage_contract_path(exp_dir, "1b")
            )
            expected_upstream = str(
                contract.get("settings", {}).get(
                    "upstream_1b_digest"
                )
                or ""
            )
            current_upstream = str(
                (upstream_1b or {}).get("digest")
                or ""
            )
            if not expected_upstream or expected_upstream != current_upstream:
                return (
                    "1C semantic token이 현재 1B HuBERT 전처리와 "
                    "연결되어 있지 않습니다. 1C를 다시 실행하세요."
                )

        contracts.append(contract)

    return None


def _training_artifact_error(exp_dir, mode, version_name):
    exp_dir = Path(exp_dir)

    def file_ok(path):
        return path.is_file() and path.stat().st_size > 0

    def dir_ok(path, allow_empty=False):
        if not path.is_dir():
            return False
        return allow_empty or any(path.iterdir())

    def text_names(path):
        names = set()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    names.add(line.split("\t", 1)[0])
        return names

    text_path = exp_dir / "2-name2text.txt"
    if not file_ok(text_path):
        return f"텍스트 전처리 산출물이 없거나 비었습니다: {text_path}"
    names_text = text_names(text_path)
    if not names_text:
        return f"텍스트 학습 샘플이 없습니다: {text_path}"

    if mode == "gpt":
        bert_dir = exp_dir / "3-bert"
        semantic_path = exp_dir / "6-name2semantic.tsv"
        if not dir_ok(bert_dir, allow_empty=True):
            return f"BERT feature 디렉터리가 없습니다: {bert_dir}"
        if not file_ok(semantic_path):
            return f"semantic token 파일이 없거나 비었습니다: {semantic_path}"

        names_semantic = set()
        with semantic_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index == 0:
                    continue
                line = line.strip()
                if line:
                    names_semantic.add(line.split("\t", 1)[0])

        if not names_semantic:
            return f"semantic token 데이터가 비었습니다: {semantic_path}"
        unknown = sorted(names_semantic - names_text)
        if unknown:
            return (
                "semantic token에 1A text가 없는 샘플이 있습니다: "
                + ", ".join(unknown[:10])
            )
        if len(names_semantic & names_text) < 2:
            return "GPT/S1 학습에 사용할 일치 샘플이 2개 미만입니다."
        return None

    if mode == "sovits":
        hubert_dir = exp_dir / "4-cnhubert"
        wav_dir = exp_dir / "5-wav32k"
        if not dir_ok(hubert_dir):
            return f"HuBERT feature가 없거나 비었습니다: {hubert_dir}"
        if not dir_ok(wav_dir):
            return f"32kHz 학습 음성이 없거나 비었습니다: {wav_dir}"

        names_hubert = {
            path.stem
            for path in hubert_dir.iterdir()
            if path.is_file() and path.suffix == ".pt"
        }
        names_wav = {
            path.name
            for path in wav_dir.iterdir()
            if path.is_file()
        }
        if names_hubert != names_wav:
            only_hubert = sorted(names_hubert - names_wav)
            only_wav = sorted(names_wav - names_hubert)
            return (
                "1B HuBERT/wav32k 샘플 집합이 다릅니다. "
                f"HuBERT-only={only_hubert[:5]}, "
                f"wav-only={only_wav[:5]}"
            )

        usable = names_text & names_hubert & names_wav
        if len(usable) < 2:
            return "SoVITS 학습에 사용할 일치 샘플이 2개 미만입니다."

        if "Pro" in str(version_name):
            sv_dir = exp_dir / "7-sv"
            if not dir_ok(sv_dir):
                return f"speaker feature가 없거나 비었습니다: {sv_dir}"
            names_sv = {
                path.stem
                for path in sv_dir.iterdir()
                if path.is_file() and path.suffix == ".pt"
            }
            missing_sv = sorted(usable - names_sv)
            if missing_sv:
                return (
                    "speaker feature가 없는 학습 샘플이 있습니다: "
                    + ", ".join(missing_sv[:10])
                )
        return None

    return f"알 수 없는 학습 모드: {mode}"



def _emotion_dataset_slug(dataset_id):
    value = str(dataset_id or "").strip() or "emotion_dataset_01"
    return (
        re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._-")
        or "emotion_dataset_01"
    )[:80]


def _emotion_output_dir(dataset_id):
    return (
        Path(now_dir)
        / "output"
        / "emotion_opt"
        / _emotion_dataset_slug(dataset_id)
    ).resolve()



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


def _stream_emotion_command(
    cmd,
    log_lines,
    stage,
    cuda_visible_devices=None,
):
    global p_emotion_training
    global emotion_task_cancel_requested

    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    child_env.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    child_env.setdefault("GLOG_minloglevel", "2")
    if cuda_visible_devices is not None:
        child_env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    command_text = " ".join(repr(str(item)) for item in cmd)
    log_lines.append(f"$ {command_text}")
    print(f"[Emotion] {command_text}", flush=True)

    emotion_task_cancel_requested = False

    try:
        process = Popen(
            cmd,
            cwd=now_dir,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        p_emotion_training = None
        raise EmotionPipelineError(
            f"{stage} failed to start: {exc}"
        ) from exc

    p_emotion_training = process

    if process.stdout is not None:
        for line in process.stdout:
            clean_line = line.rstrip()
            log_lines.append(clean_line)
            print(f"[Emotion] {clean_line}", flush=True)
            yield _emotion_status(stage, log_lines)

    return_code = process.wait()
    cancelled = emotion_task_cancel_requested

    p_emotion_training = None
    emotion_task_cancel_requested = False

    if cancelled:
        raise EmotionTaskCancelled(stage)

    if return_code != 0:
        raise EmotionPipelineError(
            f"{stage} failed with exit code {return_code}"
        )


def stop_emotion_training():
    global p_emotion_training
    global emotion_task_cancel_requested

    if p_emotion_training is None:
        return "### 감정 작업\n\n실행 중인 작업이 없습니다."

    emotion_task_cancel_requested = True

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
    dataset_id,
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
        source_list = _clean_user_path(source_list)
        wav_root = _clean_user_path(wav_root)
        sidecar_output = _clean_user_path(sidecar_output)
        model_name = str(model_name or "").strip()
        device = str(device or "auto").strip() or "auto"

        if not source_list:
            raise EmotionPipelineError(
                "먼저 GPT-SoVITS 텍스트 표기(.list) 파일을 지정하세요."
            )

        if not sidecar_output:
            sidecar_output = str(
                _emotion_output_dir(dataset_id)
                / "emotion_sidecar.jsonl"
            )

        output_path = Path(
            str(sidecar_output)
        ).expanduser().resolve()

        if not model_name:
            model_name = "iic/emotion2vec_plus_large"

        min_confidence = float(min_confidence)

        if not 0.0 <= min_confidence <= 1.0:
            raise EmotionPipelineError(
                "감정 신뢰도 기준은 0~1 사이여야 합니다."
            )

        output_before = _file_signature(output_path)

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
            "화자 기준 상대 감정 분석",
        ):
            yield (
                status,
                {"__type__": "update"},
            )

        output_after = _file_signature(output_path)
        if output_after is None:
            raise EmotionPipelineError(
                f"자동 sidecar 파일이 생성되지 않았습니다: {output_path}"
            )
        if output_before is not None and output_after == output_before:
            raise EmotionPipelineError(
                "자동 sidecar subprocess가 성공 종료됐지만 "
                "출력 파일이 갱신되지 않았습니다."
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
                f"상대 라벨 {report.get('accepted_counts', {})}"
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

    except EmotionTaskCancelled:
        p_emotion_training = None
        yield (
            _emotion_status(
                "자동 감정 데이터 생성 중지됨",
                log_lines,
            ),
            {"__type__": "update"},
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
    experiment_name,
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
        dataset_slug = _emotion_dataset_slug(dataset_id)
        emotion_output_dir = _emotion_output_dir(dataset_id)

        dataset_dir_text = _clean_user_path(dataset_dir)
        if not dataset_dir_text:
            dataset_dir_text = str(emotion_output_dir / "dataset")

        feature_root_text = _clean_user_path(feature_root)
        if not feature_root_text:
            feature_root_text = str(emotion_output_dir / "features")

        controller_output_text = _clean_user_path(controller_output)
        if not controller_output_text:
            controller_output_text = str(
                emotion_output_dir
                / f"{dataset_slug}_emotion_controller.pt"
            )

        dataset_dir = Path(dataset_dir_text).expanduser().resolve()
        feature_root = Path(feature_root_text).expanduser().resolve()
        controller_output = Path(
            controller_output_text
        ).expanduser().resolve()

        if (
            os.name != "nt"
            and str(controller_output).startswith("/content/drive/")
            and not os.path.ismount("/content/drive")
        ):
            raise EmotionPipelineError(
                "Controller 저장 경로가 Google Drive 아래이지만 "
                "/content/drive가 실제로 mount되어 있지 않습니다. "
                "Drive를 다시 mount하거나 로컬 저장 경로를 지정하세요."
            )

        metadata_path = dataset_dir / "metadata.jsonl"
        train_list = dataset_dir / "train.list"
        reference_map_path = dataset_dir / "reference_map.json"
        reference_prompt_path = dataset_dir / "reference_prompts.pt"
        tts_config_path = dataset_dir / "tts_infer_emotion_v5dev.yaml"

        s1_checkpoint_path = _clean_user_path(
            s1_checkpoint_path
        )
        experiment_name = str(experiment_name or "").strip()

        if not s1_checkpoint_path:
            auto_checkpoint = latest_gpt_checkpoint_for_experiment(
                Path(GPT_weight_version2root["v5dev"]),
                experiment_name,
            )
            if auto_checkpoint is not None:
                s1_checkpoint_path = str(auto_checkpoint)
                log_lines.append(
                    "[s1] auto-selected latest fine-tuned GPT/S1: "
                    f"{auto_checkpoint}"
                )
            else:
                s1_checkpoint_path = (
                    pretrained_gpt_name.get("v5dev")
                    or "GPT_SoVITS/pretrained_models/s1v3.ckpt"
                )
                log_lines.append(
                    "[s1] no fine-tuned GPT/S1 found for "
                    f"exp_name={experiment_name!r}; using base: "
                    f"{s1_checkpoint_path}"
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

        raw_requested_emotions = [
            item.strip().lower()
            for item in str(emotion_names).split(",")
            if item.strip()
        ]
        requested_emotions = [
            emotion
            for emotion in raw_requested_emotions
            if emotion not in {"neutral", "default"}
        ]

        if any(
            emotion in {"neutral", "default"}
            for emotion in raw_requested_emotions
        ):
            log_lines.append(
                "[emotions] neutral/default is the unmodified base "
                "and is not a controller ID"
            )

        if len(set(requested_emotions)) != len(requested_emotions):
            raise EmotionPipelineError(
                "감정 이름이 중복되어 있습니다."
            )

        if requested_emotions and len(requested_emotions) < 2:
            raise EmotionPipelineError(
                "controller 감정을 직접 지정할 경우 "
                "비-neutral 감정이 최소 2개 필요합니다."
            )

        steps = int(steps)

        if steps <= 0:
            raise EmotionPipelineError(
                "학습 step은 1 이상이어야 합니다."
            )

        if torch.cuda.is_available():
            gpu_number = str(fix_gpu_number(gpu_number)).strip()
            try:
                gpu_index = int(gpu_number)
            except ValueError as exc:
                raise EmotionPipelineError(
                    "감정 controller 학습은 현재 GPU 번호 하나만 지원합니다."
                ) from exc
            if gpu_index not in set_gpu_numbers:
                raise EmotionPipelineError(
                    f"사용할 수 없는 GPU 번호입니다: {gpu_index}"
                )
            gpu_number = str(gpu_index)
        else:
            gpu_number = "0"

        source_list = _clean_user_path(source_list)
        sidecar_path = _clean_user_path(sidecar_path)
        wav_root = _clean_user_path(wav_root)

        if not source_list:
            raise EmotionPipelineError(
                "원본 GPT-SoVITS .list 파일 경로를 입력하세요."
            )

        if not sidecar_path:
            raise EmotionPipelineError(
                "감정 controller 학습에는 sidecar JSONL이 필요합니다."
            )

        dataset_ready = (
            metadata_path.is_file()
            and train_list.is_file()
            and (dataset_dir / "report.json").is_file()
        )

        if reuse_dataset and dataset_ready:
            report = json.loads(
                (dataset_dir / "report.json").read_text(
                    encoding="utf-8"
                )
            )
            provenance = report.get("source_provenance") or {}
            current_fingerprint = build_dataset_fingerprint(
                source_list,
                wav_root or None,
            )
            current_sidecar = Path(sidecar_path).expanduser().resolve()
            current_sidecar_sha = (
                file_sha256(current_sidecar)
                if current_sidecar.is_file()
                else None
            )
            if (
                provenance.get("dataset_digest")
                != current_fingerprint.get("digest")
                or provenance.get("sidecar_sha256")
                != current_sidecar_sha
            ):
                raise EmotionPipelineError(
                    "기존 canonical dataset의 원본 .list/WAV/sidecar가 "
                    "현재 입력과 다릅니다. 재사용을 끄고 다시 생성하세요."
                )

            log_lines.append(
                f"[reuse] canonical dataset: {dataset_dir}"
            )
            yield _emotion_status(
                "1/5 감정 데이터셋 검사",
                log_lines,
            )
        else:

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

            try:
                for status in _stream_emotion_command(
                    cmd,
                    log_lines,
                    "1/5 감정 데이터셋 생성",
                ):
                    yield status
            except (EmotionTaskCancelled, EmotionPipelineError):
                if dataset_dir.is_dir():
                    shutil.rmtree(dataset_dir, ignore_errors=True)
                raise

        report = validate_emotion_dataset(dataset_dir)

        report_dataset_id = str(
            report.get("dataset_id") or ""
        ).strip()
        if report_dataset_id != dataset_id:
            raise EmotionPipelineError(
                "canonical dataset ID mismatch: "
                f"expected {dataset_id!r}, got {report_dataset_id!r}"
            )

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
                    "사용 가능한 학습 라벨이 없는 감정: "
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
            trainable_emotions=parsed_emotions,
        )
        log_lines.append(
            f"[reference] {reference_map_path}"
        )

        yield _emotion_status(
            "2/5 기준 reference 자동 선택",
            log_lines,
            "3~10초 train 샘플 중 neutral/default를 우선해 reference를 선택했습니다.",
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
            "--s1-checkpoint",
            str(s1_checkpoint),
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
            raise EmotionPipelineError(
                "V5 preprocessing succeeded but its exact build directory "
                "could not be identified from subprocess output. "
                "Refusing to guess a different build from the output root."
            )

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
            cuda_visible_devices=gpu_number,
        ):
            yield status

        controller_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        controller_before = _file_signature(controller_output)

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
            "--version",
            "v5dev",
            "--device",
            "cuda" if torch.cuda.is_available() else "cpu",
        ]

        for status in _stream_emotion_command(
            train_cmd,
            log_lines,
            "5/5 emotion controller 학습",
            cuda_visible_devices=gpu_number,
        ):
            yield status

        controller_after = _file_signature(controller_output)
        if controller_after is None:
            raise EmotionPipelineError(
                "학습 프로세스는 종료됐지만 controller 파일이 없습니다: "
                f"{controller_output}"
            )
        if (
            controller_before is not None
            and controller_after == controller_before
        ):
            raise EmotionPipelineError(
                "controller subprocess가 성공 종료됐지만 "
                "출력 체크포인트가 갱신되지 않았습니다."
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

    except EmotionTaskCancelled:
        p_emotion_training = None
        yield _emotion_status(
            "감정 제어 학습 중지됨",
            log_lines,
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

    if p_label is not None and p_label.poll() is None:
        yield (
            process_info(process_name_subfix, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        return

    p_label = None
    list_path = _resolve_repo_path(path_list)

    if list_path is None or not list_path.is_file():
        yield (
            process_info(process_name_subfix, "failed")
            + f": list file not found: {list_path or '<empty>'}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    script_path = _resolve_repo_path("tools/subfix_webui.py")
    if script_path is None or not script_path.is_file():
        yield (
            process_info(process_name_subfix, "failed")
            + f": script not found: {script_path}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    cmd = [
        python_exec,
        "-s",
        str(script_path),
        "--load_list",
        str(list_path),
        "--webui_port",
        str(webui_port_subfix),
        "--is_share",
        "True",
    ]
    print("[Label WebUI]", " ".join(repr(item) for item in cmd))

    try:
        process = Popen(
            cmd,
            cwd=now_dir,
        )
        p_label = process
    except OSError as exc:
        p_label = None
        yield (
            process_info(process_name_subfix, "failed") + f": {exc}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    yield (
        process_info(process_name_subfix, "opened"),
        {"__type__": "update", "visible": False},
        {"__type__": "update", "visible": True},
    )


def close_label_webui():
    global p_label

    if p_label is not None and p_label.poll() is None:
        kill_process(p_label.pid, process_name_subfix)

    p_label = None

    return (
        process_info(process_name_subfix, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


process_name_uvr5 = i18n("人声分离WebUI")


process_name_uvr5 = i18n("人声分离WebUI")


def change_uvr5():
    global p_uvr5

    if p_uvr5 is not None and p_uvr5.poll() is None:
        yield (
            process_info(process_name_uvr5, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        return

    p_uvr5 = None
    script_path = _resolve_repo_path("tools/uvr5/webui.py")

    if script_path is None or not script_path.is_file():
        yield (
            process_info(process_name_uvr5, "failed")
            + f": script not found: {script_path}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    cmd = [
        python_exec,
        "-s",
        str(script_path),
        str(infer_device),
        str(is_half),
        str(webui_port_uvr5),
        "True",
    ]
    print("[UVR5]", " ".join(repr(item) for item in cmd))

    try:
        process = Popen(
            cmd,
            cwd=now_dir,
        )
        p_uvr5 = process
    except OSError as exc:
        p_uvr5 = None
        yield (
            process_info(process_name_uvr5, "failed") + f": {exc}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    yield (
        process_info(process_name_uvr5, "opened"),
        {"__type__": "update", "visible": False},
        {"__type__": "update", "visible": True},
    )


def close_uvr5_webui():
    global p_uvr5

    if p_uvr5 is not None and p_uvr5.poll() is None:
        kill_process(p_uvr5.pid, process_name_uvr5)

    p_uvr5 = None

    return (
        process_info(process_name_uvr5, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


process_name_tts = i18n("TTS推理WebUI")


process_name_tts = i18n("TTS推理WebUI")


def _stream_tts_inference_output(process, allow_local_url=False):
    global tts_inference_url

    if process.stdout is None:
        return

    public_url_pattern = re.compile(
        r"https://[A-Za-z0-9-]+\.gradio\.live"
    )
    local_url_pattern = re.compile(
        r"https?://(?:127\.0\.0\.1|localhost|0\.0\.0\.0):\d+"
    )

    for line in process.stdout:
        clean_line = line.rstrip()
        print(f"[TTS] {clean_line}", flush=True)
        tts_inference_logs.append(clean_line)

        match = public_url_pattern.search(clean_line)
        if match is None and allow_local_url:
            match = local_url_pattern.search(clean_line)

        if match is not None:
            tts_inference_url = match.group(0).replace(
                "://0.0.0.0:",
                "://127.0.0.1:",
            )


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
    global tts_inference_cancel_requested

    if p_tts_inference is not None:
        if p_tts_inference.poll() is None:
            yield (
                process_info(process_name_tts, "occupy"),
                {"__type__": "update", "visible": False},
                {"__type__": "update", "visible": True},
            )
            return
        p_tts_inference = None

    gpt_value = name2gpt_path.get(gpt_path, gpt_path)
    sovits_value = name2sovits_path.get(sovits_path, sovits_path)

    resolved = {
        "GPT/S1": (_resolve_repo_path(gpt_value), "file"),
        "SoVITS/S2": (_resolve_repo_path(sovits_value), "file"),
        "BERT": (_resolve_repo_path(bert_path), "dir"),
        "CNHuBERT": (_resolve_repo_path(cnhubert_base_path), "dir"),
    }

    invalid_paths = []
    for label, (path, kind) in resolved.items():
        valid = (
            path is not None
            and (
                path.is_file()
                if kind == "file"
                else path.is_dir()
            )
        )
        if not valid:
            invalid_paths.append(f"{label}: {path or '<empty>'}")

    if invalid_paths:
        yield (
            "### TTS 추론 시작 실패\n\n잘못된 모델 경로:\n- "
            + "\n- ".join(invalid_paths),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    gpt_path = str(resolved["GPT/S1"][0])
    sovits_path = str(resolved["SoVITS/S2"][0])
    bert_path = str(resolved["BERT"][0])
    cnhubert_base_path = str(resolved["CNHuBERT"][0])

    controller_text = _clean_user_path(
        latest_emotion_controller_path
    )
    controller_path = (
        _resolve_repo_path(controller_text)
        if controller_text
        else None
    )

    if controller_text and (
        controller_path is None
        or not controller_path.is_file()
    ):
        yield (
            "### TTS 추론 시작 실패\n\n"
            f"감정 controller 파일이 없습니다: {controller_path}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    controller_available = controller_path is not None

    script_path = (
        "GPT_SoVITS/inference_webui.py"
        if controller_available
        else (
            "GPT_SoVITS/inference_webui_fast.py"
            if batched_infer_enabled
            else "GPT_SoVITS/inference_webui.py"
        )
    )
    script_file = (Path(now_dir) / script_path).resolve()

    if not script_file.is_file():
        yield (
            f"### TTS 추론 시작 실패\n\n스크립트가 없습니다: {script_file}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    cmd = [
        python_exec,
        "-u",
        "-s",
        str(script_file),
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
            "is_share": "True",
            "version": str(version),
            "language": str(language),
            "emotion_controller_path": (
                str(controller_path)
                if controller_available
                else ""
            ),
            "PYTHONUNBUFFERED": "1",
        }
    )

    tts_inference_url = None
    tts_inference_logs.clear()
    tts_inference_cancel_requested = False

    print(
        "[TTS] launch:",
        " ".join(repr(item) for item in cmd),
        flush=True,
    )

    try:
        process = Popen(
            cmd,
            cwd=now_dir,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except (OSError, TypeError) as exc:
        yield (
            f"### TTS 추론 시작 실패\n\n{exc}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    p_tts_inference = process

    tts_inference_reader = threading.Thread(
        target=_stream_tts_inference_output,
        args=(process, False),
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
            p_tts_inference = None

            if tts_inference_cancel_requested:
                tts_inference_cancel_requested = False
                return

            log_tail = _tts_log_tail()
            message = (
                f"### {i18n('TTS推理WebUI启动失败')} "
                f"(exit code {return_code})"
            )

            if log_tail:
                message += "\n\n" + log_tail

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
        message += "\n\n" + log_tail

    yield (
        message,
        {"__type__": "update", "visible": False},
        {"__type__": "update", "visible": True},
    )


def close_tts_inference():
    global p_tts_inference
    global tts_inference_url
    global tts_inference_cancel_requested

    tts_inference_cancel_requested = True

    if (
        p_tts_inference is not None
        and p_tts_inference.poll() is None
    ):
        kill_process(p_tts_inference.pid, process_name_tts)

    p_tts_inference = None
    tts_inference_url = None
    tts_inference_logs.clear()

    return (
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

    asr_inp_dir = str(asr_inp_dir or "").strip().strip(chr(39) + chr(34)).replace(chr(0x202A), "")
    asr_opt_dir = str(asr_opt_dir or "").strip().strip(chr(39) + chr(34)).replace(chr(0x202A), "")

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
    output_before = _file_signature(output_file_path)

    print("[ASR]", " ".join(repr(item) for item in cmd), flush=True)
    asr_cancel_requested = False

    try:
        process = Popen(
            cmd,
            cwd=now_dir,
        )
        p_asr = process
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

    yield (
        process_info(process_name_asr, "opened"),
        {"__type__": "update", "visible": False},
        {"__type__": "update", "visible": True},
        {"__type__": "update"},
        {"__type__": "update"},
        {"__type__": "update"},
    )

    try:
        return_code = process.wait()
    finally:
        if p_asr is process:
            p_asr = None

    if asr_cancel_requested:
        asr_cancel_requested = False
        return

    output_after = _file_signature(output_file_path)

    if (
        return_code != 0
        or output_after is None
        or (
            output_before is not None
            and output_after == output_before
        )
    ):
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
        if p_asr.poll() is None:
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
    global denoise_cancel_requested

    if p_denoise is not None:
        yield (
            process_info(process_name_denoise, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    input_path = _resolve_repo_path(denoise_inp_dir)
    if input_path is None or not input_path.is_dir():
        yield (
            process_info(process_name_denoise, "failed")
            + f": input directory not found: {input_path or '<empty>'}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    output_text = _clean_user_path(denoise_opt_dir)
    if not output_text:
        output_text = "output/denoise_opt"
    output_path = _resolve_repo_path(output_text)
    if output_path is None:
        yield (
            process_info(process_name_denoise, "failed")
            + ": invalid output directory",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    cmd = [
        python_exec,
        "-s",
        "tools/cmd-denoise.py",
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-p",
        "float16" if is_half else "float32",
    ]

    print("[Denoise]", " ".join(repr(item) for item in cmd))
    denoise_cancel_requested = False

    try:
        process = Popen(
            cmd,
            cwd=now_dir,
        )
        p_denoise = process
    except OSError as exc:
        p_denoise = None
        yield (
            process_info(process_name_denoise, "failed") + f": {exc}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    yield (
        process_info(process_name_denoise, "opened"),
        {"__type__": "update", "visible": False},
        {"__type__": "update", "visible": True},
        {"__type__": "update"},
        {"__type__": "update"},
    )

    return_code = process.wait()
    if p_denoise is process:
        p_denoise = None

    if denoise_cancel_requested:
        denoise_cancel_requested = False
        return

    if return_code != 0:
        yield (
            process_info(process_name_denoise, "failed")
            + f" (exit code {return_code})",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    if (
        not output_path.is_dir()
        or not any(path.is_file() for path in output_path.rglob("*"))
    ):
        yield (
            process_info(process_name_denoise, "failed")
            + ": no denoised files were produced",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    yield (
        process_info(process_name_denoise, "finish"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
        {"__type__": "update", "value": str(output_path)},
        {"__type__": "update", "value": str(output_path)},
    )


def close_denoise():
    global p_denoise
    global denoise_cancel_requested

    denoise_cancel_requested = True

    if (
        p_denoise is not None
        and p_denoise.poll() is None
    ):
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
    inp_text,
    inp_wav_dir,
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
    global sovits_train_cancel_requested

    if p_train_SoVITS is not None:
        yield (
            process_info(process_name_sovits, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    exp_name = str(exp_name or "").strip()
    if not exp_name:
        yield (
            process_info(process_name_sovits, "failed")
            + ": experiment name is empty",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    s2_dir = Path(exp_root) / exp_name
    artifact_error = _training_artifact_error(
        s2_dir,
        "sovits",
        version,
    )
    if artifact_error:
        yield (
            process_info(process_name_sovits, "failed")
            + f": {artifact_error}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    dataset_error = _validate_training_dataset_contract(
        s2_dir,
        inp_text,
        inp_wav_dir,
        version,
        ("1a", "1b"),
    )
    if dataset_error:
        yield (
            process_info(process_name_sovits, "failed")
            + f": {dataset_error}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    config_file = (
        "GPT_SoVITS/configs/s2.json"
        if version not in {"v2Pro", "v2ProPlus"}
        else f"GPT_SoVITS/configs/s2{version}.json"
    )
    config_path = _resolve_repo_path(config_file)
    if config_path is None or not config_path.is_file():
        yield (
            process_info(process_name_sovits, "failed")
            + f": config not found: {config_path}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    pretrained_s2g_path = _resolve_repo_path(pretrained_s2G)
    if pretrained_s2g_path is None or not pretrained_s2g_path.is_file():
        yield (
            process_info(process_name_sovits, "failed")
            + f": pretrained S2G not found: {pretrained_s2g_path}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return
    pretrained_s2G = str(pretrained_s2g_path)

    pretrained_s2d_path = None
    if pretrained_s2D:
        pretrained_s2d_path = _resolve_repo_path(pretrained_s2D)
        if (
            version not in v3v4set
            and (
                pretrained_s2d_path is None
                or not pretrained_s2d_path.is_file()
            )
        ):
            yield (
                process_info(process_name_sovits, "failed")
                + f": pretrained S2D not found: {pretrained_s2d_path}",
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
                {"__type__": "update"},
                {"__type__": "update"},
            )
            return
        if pretrained_s2d_path is not None:
            pretrained_s2D = str(pretrained_s2d_path)

    contract_1a = read_stage_contract(
        stage_contract_path(s2_dir, "1a")
    )
    contract_1b = read_stage_contract(
        stage_contract_path(s2_dir, "1b")
    )
    if version in v3v4set:
        resume_dir = (
            s2_dir / f"logs_s2_{version}_lora_{int(lora_rank)}"
        )
    else:
        resume_dir = s2_dir / f"logs_s2_{version}"

    resume_identity = {
        "version": str(version),
        "preprocess_1a": str((contract_1a or {}).get("digest") or ""),
        "preprocess_1b": str((contract_1b or {}).get("digest") or ""),
        "pretrained_s2g": str(pretrained_s2g_path),
        "pretrained_s2g_signature": quick_path_signature(
            pretrained_s2g_path
        ),
        "pretrained_s2d": (
            str(pretrained_s2d_path)
            if pretrained_s2d_path is not None
            else None
        ),
        "pretrained_s2d_signature": (
            quick_path_signature(pretrained_s2d_path)
            if pretrained_s2d_path is not None
            else None
        ),
        "s2_config_signature": quick_path_signature(config_path),
        "lora_rank": (
            int(lora_rank)
            if version in v3v4set
            else None
        ),
    }
    resume_marker_name = (
        f"training_resume_s2_{version}_lora_{int(lora_rank)}.json"
        if version in v3v4set
        else f"training_resume_s2_{version}.json"
    )
    resume_reset = prepare_training_resume_state(
        s2_dir / resume_marker_name,
        [resume_dir],
        resume_identity,
    )
    if resume_reset:
        print(
            "[SoVITS resume] incompatible previous full checkpoint state "
            "was cleared; exported SoVITS_weights are untouched"
        )

    try:
        data = json.loads(
            config_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        yield (
            process_info(process_name_sovits, "failed") + f": {exc}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    (s2_dir / f"logs_s2_{version}").mkdir(
        parents=True,
        exist_ok=True,
    )

    if is_half is False:
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
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = str(s2_dir)
    data["save_weight_dir"] = SoVITS_weight_version2root[version]
    data["name"] = exp_name
    data["version"] = version

    tmp_config_path = Path(tmp) / "tmp_s2.json"
    tmp_config_path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    script = (
        "GPT_SoVITS/s2_train.py"
        if version in ["v1", "v2", "v2Pro", "v2ProPlus"]
        else "GPT_SoVITS/s2_train_v3_lora.py"
    )
    cmd = [
        python_exec,
        "-s",
        script,
        "--config",
        str(tmp_config_path),
    ]
    print("[SoVITS train]", " ".join(repr(x) for x in cmd))
    sovits_train_cancel_requested = False

    try:
        process = Popen(
            cmd,
            cwd=now_dir,
        )
        p_train_SoVITS = process
    except OSError as exc:
        p_train_SoVITS = None
        yield (
            process_info(process_name_sovits, "failed") + f": {exc}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    yield (
        process_info(process_name_sovits, "opened"),
        {"__type__": "update", "visible": False},
        {"__type__": "update", "visible": True},
        {"__type__": "update"},
        {"__type__": "update"},
    )

    return_code = process.wait()
    if p_train_SoVITS is process:
        p_train_SoVITS = None

    if sovits_train_cancel_requested:
        sovits_train_cancel_requested = False
        return

    if return_code != 0:
        yield (
            process_info(process_name_sovits, "failed")
            + f" (exit code {return_code})",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    SoVITS_dropdown_update, GPT_dropdown_update = change_choices()
    yield (
        process_info(process_name_sovits, "finish"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
        SoVITS_dropdown_update,
        GPT_dropdown_update,
    )


def close1Ba():
    global p_train_SoVITS
    global sovits_train_cancel_requested

    sovits_train_cancel_requested = True

    if (
        p_train_SoVITS is not None
        and p_train_SoVITS.poll() is None
    ):
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
    inp_text,
    inp_wav_dir,
    if_dpo,
    if_save_latest,
    if_save_every_weights,
    save_every_epoch,
    gpu_numbers,
    pretrained_s1,
    pretrained_s2G_path,
    use_multilingual_bert,
    multilingual_bert_model,
):
    global p_train_GPT
    global gpt_train_cancel_requested

    if p_train_GPT is not None:
        yield (
            process_info(process_name_gpt, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    exp_name = str(exp_name or "").strip()
    if not exp_name:
        yield (
            process_info(process_name_gpt, "failed")
            + ": experiment name is empty",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    s1_dir = Path(exp_root) / exp_name
    artifact_error = _training_artifact_error(
        s1_dir,
        "gpt",
        version,
    )
    if artifact_error:
        yield (
            process_info(process_name_gpt, "failed")
            + f": {artifact_error}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    dataset_error = _validate_training_dataset_contract(
        s1_dir,
        inp_text,
        inp_wav_dir,
        version,
        ("1a", "1c"),
    )
    if dataset_error:
        yield (
            process_info(process_name_gpt, "failed")
            + f": {dataset_error}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    semantic_contract = read_stage_contract(
        stage_contract_path(s1_dir, "1c")
    )
    semantic_settings = (semantic_contract or {}).get("settings", {})
    current_s2_path = _resolve_repo_path(pretrained_s2G_path)
    if current_s2_path is None or not current_s2_path.is_file():
        yield (
            process_info(process_name_gpt, "failed")
            + f": S2 checkpoint not found: {current_s2_path}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return
    if (
        str(semantic_settings.get("pretrained_s2G") or "")
        != str(current_s2_path)
        or str(
            semantic_settings.get("pretrained_s2G_signature") or ""
        )
        != str(_file_signature(current_s2_path))
    ):
        yield (
            process_info(process_name_gpt, "failed")
            + ": 1C semantic tokens were generated with a different "
            "S2 checkpoint. Re-run 1C with the selected S2.",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    expected_text_features = _text_feature_config(
        use_multilingual_bert,
        multilingual_bert_model,
    )
    feature_marker = s1_dir / "text_feature_config.json"
    if feature_marker.is_file():
        try:
            actual_text_features = json.loads(
                feature_marker.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            yield (
                process_info(process_name_gpt, "failed")
                + f": invalid text feature config: {exc}",
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
                {"__type__": "update"},
                {"__type__": "update"},
            )
            return
    else:
        actual_text_features = _text_feature_config(
            False,
            multilingual_bert_model,
        )

    if actual_text_features != expected_text_features:
        yield (
            process_info(process_name_gpt, "failed")
            + ": 1A text feature mode and GPT training mode do not match",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return


    config_path = _resolve_repo_path(
        "GPT_SoVITS/configs/s1longer.yaml"
        if version == "v1"
        else "GPT_SoVITS/configs/s1longer-v2.yaml"
    )
    if config_path is None or not config_path.is_file():
        yield (
            process_info(process_name_gpt, "failed")
            + f": config not found: {config_path}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    pretrained_s1_path = _resolve_repo_path(pretrained_s1)
    if pretrained_s1_path is None or not pretrained_s1_path.is_file():
        yield (
            process_info(process_name_gpt, "failed")
            + f": pretrained S1 not found: {pretrained_s1_path}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return
    pretrained_s1 = str(pretrained_s1_path)

    contract_1a = read_stage_contract(
        stage_contract_path(s1_dir, "1a")
    )
    contract_1c = read_stage_contract(
        stage_contract_path(s1_dir, "1c")
    )
    resume_identity = {
        "version": str(version),
        "preprocess_1a": str((contract_1a or {}).get("digest") or ""),
        "preprocess_1c": str((contract_1c or {}).get("digest") or ""),
        "pretrained_s1": str(pretrained_s1_path),
        "pretrained_s1_signature": quick_path_signature(
            pretrained_s1_path
        ),
        "s1_config_signature": quick_path_signature(config_path),
        "if_dpo": bool(if_dpo),
    }
    resume_reset = prepare_training_resume_state(
        s1_dir / f"training_resume_s1_{version}.json",
        [s1_dir / f"logs_s1_{version}" / "ckpt"],
        resume_identity,
    )
    if resume_reset:
        print(
            "[GPT resume] incompatible previous full checkpoint state "
            "was cleared; exported GPT_weights are untouched"
        )

    try:
        data = yaml.load(
            config_path.read_text(encoding="utf-8"),
            Loader=yaml.FullLoader,
        )
    except (OSError, yaml.YAMLError) as exc:
        yield (
            process_info(process_name_gpt, "failed") + f": {exc}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    (s1_dir / "logs_s1").mkdir(
        parents=True,
        exist_ok=True,
    )

    if is_half is False:
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
    data["text_features"] = expected_text_features
    data["train_semantic_path"] = str(
        s1_dir / "6-name2semantic.tsv"
    )
    data["train_phoneme_path"] = str(
        s1_dir / "2-name2text.txt"
    )
    data["output_dir"] = str(
        s1_dir / f"logs_s1_{version}"
    )

    child_env = os.environ.copy()
    child_env["_CUDA_VISIBLE_DEVICES"] = str(
        fix_gpu_numbers(gpu_numbers.replace("-", ","))
    )
    child_env["hz"] = "25hz"

    tmp_config_path = Path(tmp) / "tmp_s1.yaml"
    tmp_config_path.write_text(
        yaml.dump(data, default_flow_style=False),
        encoding="utf-8",
    )

    cmd = [
        python_exec,
        "-s",
        "GPT_SoVITS/s1_train.py",
        "--config_file",
        str(tmp_config_path),
    ]
    print("[GPT train]", " ".join(repr(x) for x in cmd))
    gpt_train_cancel_requested = False

    try:
        process = Popen(
            cmd,
            cwd=now_dir,
            env=child_env,
        )
        p_train_GPT = process
    except OSError as exc:
        p_train_GPT = None
        yield (
            process_info(process_name_gpt, "failed") + f": {exc}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    yield (
        process_info(process_name_gpt, "opened"),
        {"__type__": "update", "visible": False},
        {"__type__": "update", "visible": True},
        {"__type__": "update"},
        {"__type__": "update"},
    )

    return_code = process.wait()
    if p_train_GPT is process:
        p_train_GPT = None

    if gpt_train_cancel_requested:
        gpt_train_cancel_requested = False
        return

    if return_code != 0:
        yield (
            process_info(process_name_gpt, "failed")
            + f" (exit code {return_code})",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    SoVITS_dropdown_update, GPT_dropdown_update = change_choices()
    yield (
        process_info(process_name_gpt, "finish"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
        SoVITS_dropdown_update,
        GPT_dropdown_update,
    )


def close1Bb():
    global p_train_GPT
    global gpt_train_cancel_requested

    gpt_train_cancel_requested = True

    if (
        p_train_GPT is not None
        and p_train_GPT.poll() is None
    ):
        kill_process(p_train_GPT.pid, process_name_gpt)

    p_train_GPT = None

    return (
        process_info(process_name_gpt, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


training_sequence_cancel_requested = False
process_name_train_sequence = "GPT → SoVITS 연속 학습"


def run_gpt_then_sovits(
    batch_size_gpt,
    total_epoch_gpt,
    exp_name,
    inp_text,
    inp_wav_dir,
    if_dpo,
    if_save_latest_gpt,
    if_save_every_weights_gpt,
    save_every_epoch_gpt,
    gpu_numbers_gpt,
    pretrained_s1,
    use_multilingual_bert,
    multilingual_bert_model,
    version,
    batch_size_sovits,
    total_epoch_sovits,
    text_low_lr_rate,
    if_save_latest_sovits,
    if_save_every_weights_sovits,
    save_every_epoch_sovits,
    gpu_numbers_sovits,
    pretrained_s2G,
    pretrained_s2D,
    if_grad_ckpt,
    lora_rank,
):
    global training_sequence_cancel_requested

    training_sequence_cancel_requested = False
    sovits_update = {"__type__": "update"}
    gpt_update = {"__type__": "update"}

    yield (
        process_info(process_name_gpt, "open"),
        {"__type__": "update", "visible": False},
        {"__type__": "update", "visible": True},
        sovits_update,
        gpt_update,
    )

    gpt_finished = False
    for result in open1Bb(
        batch_size_gpt,
        total_epoch_gpt,
        exp_name,
        inp_text,
        inp_wav_dir,
        if_dpo,
        if_save_latest_gpt,
        if_save_every_weights_gpt,
        save_every_epoch_gpt,
        gpu_numbers_gpt,
        pretrained_s1,
        pretrained_s2G,
        use_multilingual_bert,
        multilingual_bert_model,
    ):
        if training_sequence_cancel_requested:
            return

        child_status = result[0] if result else ""
        if len(result) >= 4:
            sovits_update = result[2]
            gpt_update = result[3]

        gpt_finished = (
            child_status
            == process_info(process_name_gpt, "finish")
        )
        yield (
            child_status,
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            sovits_update,
            gpt_update,
        )

    if not gpt_finished:
        yield (
            process_info(process_name_gpt, "failed"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            sovits_update,
            gpt_update,
        )
        return

    if training_sequence_cancel_requested:
        return

    yield (
        process_info(process_name_sovits, "open"),
        {"__type__": "update", "visible": False},
        {"__type__": "update", "visible": True},
        sovits_update,
        gpt_update,
    )

    sovits_finished = False
    for result in open1Ba(
        version,
        batch_size_sovits,
        total_epoch_sovits,
        exp_name,
        inp_text,
        inp_wav_dir,
        text_low_lr_rate,
        if_save_latest_sovits,
        if_save_every_weights_sovits,
        save_every_epoch_sovits,
        gpu_numbers_sovits,
        pretrained_s2G,
        pretrained_s2D,
        if_grad_ckpt,
        lora_rank,
    ):
        if training_sequence_cancel_requested:
            return

        child_status = result[0] if result else ""
        if len(result) >= 4:
            sovits_update = result[2]
            gpt_update = result[3]

        sovits_finished = (
            child_status
            == process_info(process_name_sovits, "finish")
        )
        yield (
            child_status,
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            sovits_update,
            gpt_update,
        )

    if not sovits_finished:
        yield (
            process_info(process_name_sovits, "failed"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            sovits_update,
            gpt_update,
        )
        return

    yield (
        process_info(process_name_train_sequence, "finish"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
        sovits_update,
        gpt_update,
    )

def stop_gpt_then_sovits():
    global training_sequence_cancel_requested

    training_sequence_cancel_requested = True

    if (
        p_train_GPT is not None
        and p_train_GPT.poll() is None
    ):
        close1Bb()

    if (
        p_train_SoVITS is not None
        and p_train_SoVITS.poll() is None
    ):
        close1Ba()

    return (
        process_info(process_name_train_sequence, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps_slice = []
slice_cancel_requested = False
process_name_slice = i18n("语音切分")


def open_slice(
    inp,
    opt_root,
    threshold,
    min_length,
    min_interval,
    hop_size,
    max_sil_kept,
    _max,
    alpha,
    n_parts,
):
    global ps_slice
    global slice_cancel_requested

    if ps_slice != []:
        yield (
            process_info(process_name_slice, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    input_path = _resolve_repo_path(inp)
    if (
        input_path is None
        or not input_path.exists()
        or not (input_path.is_file() or input_path.is_dir())
    ):
        yield (
            i18n("输入路径不存在"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    output_text = _clean_user_path(opt_root)
    if not output_text:
        output_text = "output/slicer_opt"
    output_path = _resolve_repo_path(output_text)
    if output_path is None:
        yield (
            process_info(process_name_slice, "failed")
            + ": invalid output directory",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    try:
        n_parts = 1 if input_path.is_file() else max(1, int(n_parts))
    except (TypeError, ValueError) as exc:
        yield (
            process_info(process_name_slice, "failed") + f": {exc}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    slice_cancel_requested = False

    try:
        for i_part in range(n_parts):
            cmd = [
                python_exec,
                "-s",
                "tools/slice_audio.py",
                str(input_path),
                str(output_path),
                str(threshold),
                str(min_length),
                str(min_interval),
                str(hop_size),
                str(max_sil_kept),
                str(_max),
                str(alpha),
                str(i_part),
                str(n_parts),
            ]
            print("[Slicer]", " ".join(repr(item) for item in cmd))
            ps_slice.append(
                Popen(
                    cmd,
                    cwd=now_dir,
                )
            )
    except OSError as exc:
        for process in ps_slice:
            if process.poll() is None:
                kill_process(process.pid, process_name_slice)
        ps_slice = []
        yield (
            process_info(process_name_slice, "failed") + f": {exc}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    processes = list(ps_slice)

    yield (
        process_info(process_name_slice, "opened"),
        {"__type__": "update", "visible": False},
        {"__type__": "update", "visible": True},
        {"__type__": "update"},
        {"__type__": "update"},
        {"__type__": "update"},
    )

    return_codes = [process.wait() for process in processes]
    ps_slice = []

    if slice_cancel_requested:
        slice_cancel_requested = False
        return

    if any(code != 0 for code in return_codes):
        yield (
            process_info(process_name_slice, "failed")
            + f" (exit codes: {return_codes})",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    if (
        not output_path.is_dir()
        or not any(path.is_file() for path in output_path.rglob("*"))
    ):
        yield (
            process_info(process_name_slice, "failed")
            + ": no sliced audio files were produced",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return

    yield (
        process_info(process_name_slice, "finish"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
        {"__type__": "update", "value": str(output_path)},
        {"__type__": "update", "value": str(output_path)},
        {"__type__": "update", "value": str(output_path)},
    )


def close_slice():
    global ps_slice
    global slice_cancel_requested

    slice_cancel_requested = True

    if ps_slice != []:
        for process in list(ps_slice):
            try:
                if process.poll() is None:
                    kill_process(process.pid, process_name_slice)
            except Exception:
                traceback.print_exc()

    ps_slice = []

    return (
        process_info(process_name_slice, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps1a = []
preprocess_1a_cancel_requested = False
process_name_1a = i18n("文本分词与特征提取")


def open1a(
    inp_text,
    inp_wav_dir,
    exp_name,
    gpu_numbers,
    bert_pretrained_dir,
    use_multilingual_bert,
    multilingual_bert_model,
):
    global ps1a
    global preprocess_1a_cancel_requested

    preprocess_1a_cancel_requested = False

    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)

    if not check_for_existance(
        [inp_text, inp_wav_dir],
        is_dataset_processing=True,
    ):
        yield (
            process_info(process_name_1a, "failed"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    check_details(
        [inp_text, inp_wav_dir],
        is_dataset_processing=True,
    )

    exp_name = exp_name.rstrip(" ")

    if ps1a != []:
        yield (
            process_info(process_name_1a, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        return

    opt_dir = Path(exp_root) / exp_name
    expected_contract = _preprocess_contract(
        "1a",
        inp_text,
        inp_wav_dir,
        version,
        {
            "bert_pretrained_dir": str(
                _resolve_repo_path(bert_pretrained_dir)
                or bert_pretrained_dir
            ),
            "bert_pretrained_signature": quick_path_signature(
                _resolve_repo_path(bert_pretrained_dir)
                or bert_pretrained_dir
            ),
            "use_multilingual_bert": bool(use_multilingual_bert),
            "multilingual_bert_model": str(multilingual_bert_model),
        },
    )
    source_error = _stage_dataset_error(expected_contract)
    if source_error:
        yield (
            process_info(process_name_1a, "failed")
            + ": invalid dataset: "
            + source_error,
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    _ensure_stage_contract(opt_dir, expected_contract)

    base_config = {
        "inp_text": inp_text,
        "inp_wav_dir": inp_wav_dir,
        "exp_name": exp_name,
        "opt_dir": str(opt_dir),
        "bert_pretrained_dir": bert_pretrained_dir,
        "use_multilingual_bert": str(bool(use_multilingual_bert)),
        "multilingual_bert_model": str(multilingual_bert_model),
        "version": str(version),
    }
    gpu_names = gpu_numbers.split("-")
    all_parts = len(gpu_names)

    try:
        for i_part in range(all_parts):
            child_env = os.environ.copy()
            child_env.update(
                {
                    **base_config,
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": str(
                        fix_gpu_number(gpu_names[i_part])
                    ),
                    "is_half": str(is_half),
                }
            )
            cmd = [
                python_exec,
                "-s",
                "GPT_SoVITS/prepare_datasets/1-get-text.py",
            ]
            print("[1A]", " ".join(repr(item) for item in cmd))
            ps1a.append(
                Popen(
                    cmd,
                    cwd=now_dir,
                    env=child_env,
                )
            )
    except OSError as exc:
        for process in ps1a:
            if process.poll() is None:
                kill_process(process.pid, process_name_1a)
        ps1a = []
        yield (
            process_info(process_name_1a, "failed") + f": {exc}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    processes = list(ps1a)

    yield (
        process_info(process_name_1a, "running"),
        {"__type__": "update", "visible": False},
        {"__type__": "update", "visible": True},
    )

    return_codes = [process.wait() for process in processes]
    ps1a = []

    if preprocess_1a_cancel_requested:
        preprocess_1a_cancel_requested = False
        return

    if any(code != 0 for code in return_codes):
        yield (
            process_info(process_name_1a, "failed")
            + f" (exit codes: {return_codes})",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    opt = []
    try:
        for i_part in range(all_parts):
            txt_path = Path(opt_dir) / f"2-name2text-{i_part}.txt"
            if not txt_path.is_file():
                raise FileNotFoundError(
                    f"1A output missing: {txt_path}"
                )
            with txt_path.open("r", encoding="utf8") as handle:
                opt += handle.read().strip("\n").split("\n")
            txt_path.unlink()

        expected_rows = _nonempty_line_count(inp_text)
        if len(opt) != expected_rows:
            raise RuntimeError(
                "1A output count mismatch: "
                f"expected {expected_rows}, got {len(opt)}"
            )

        path_text = Path(opt_dir) / "2-name2text.txt"
        path_text.parent.mkdir(parents=True, exist_ok=True)
        path_text.write_text(
            "\n".join(opt) + "\n",
            encoding="utf8",
        )
        _write_text_feature_config(
            path_text,
            use_multilingual_bert,
            multilingual_bert_model,
        )
        write_stage_contract(
            stage_contract_path(opt_dir, "1a"),
            expected_contract,
        )
    except OSError as exc:
        yield (
            process_info(process_name_1a, "failed") + f": {exc}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

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


def close1a():
    global ps1a
    global preprocess_1a_cancel_requested

    preprocess_1a_cancel_requested = True

    if ps1a != []:
        for p1a in ps1a:
            try:
                if p1a.poll() is None:
                    kill_process(p1a.pid, process_name_1a)
            except Exception:
                traceback.print_exc()
        ps1a = []
    return (
        process_info(process_name_1a, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


sv_path = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
ps1b = []
preprocess_1b_cancel_requested = False
process_name_1b = i18n("语音自监督特征提取")


def open1b(
    version,
    inp_text,
    inp_wav_dir,
    exp_name,
    gpu_numbers,
    ssl_pretrained_dir,
):
    global ps1b
    global preprocess_1b_cancel_requested

    preprocess_1b_cancel_requested = False

    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)

    if not check_for_existance(
        [inp_text, inp_wav_dir],
        is_dataset_processing=True,
    ):
        yield (
            process_info(process_name_1b, "failed"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    check_details(
        [inp_text, inp_wav_dir],
        is_dataset_processing=True,
    )

    exp_name = exp_name.rstrip(" ")

    if ps1b != []:
        yield (
            process_info(process_name_1b, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        return

    opt_dir = Path(exp_root) / exp_name
    expected_contract = _preprocess_contract(
        "1b",
        inp_text,
        inp_wav_dir,
        version,
        {
            "cnhubert_base_dir": str(
                _resolve_repo_path(ssl_pretrained_dir)
                or ssl_pretrained_dir
            ),
            "cnhubert_signature": quick_path_signature(
                _resolve_repo_path(ssl_pretrained_dir)
                or ssl_pretrained_dir
            ),
            "sv_path": (
                str(_resolve_repo_path(sv_path) or sv_path)
                if "Pro" in version
                else None
            ),
            "sv_signature": (
                quick_path_signature(
                    _resolve_repo_path(sv_path) or sv_path
                )
                if "Pro" in version
                else None
            ),
        },
    )
    source_error = _stage_dataset_error(expected_contract)
    if source_error:
        yield (
            process_info(process_name_1b, "failed")
            + ": invalid dataset: "
            + source_error,
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    _ensure_stage_contract(opt_dir, expected_contract)

    base_config = {
        "inp_text": inp_text,
        "inp_wav_dir": inp_wav_dir,
        "exp_name": exp_name,
        "opt_dir": str(opt_dir),
        "cnhubert_base_dir": ssl_pretrained_dir,
        "sv_path": sv_path,
        "is_half": str(is_half),
        "version": str(version),
    }
    gpu_names = gpu_numbers.split("-")
    all_parts = len(gpu_names)

    try:
        for i_part in range(all_parts):
            child_env = os.environ.copy()
            child_env.update(
                {
                    **base_config,
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": str(
                        fix_gpu_number(gpu_names[i_part])
                    ),
                }
            )
            cmd = [
                python_exec,
                "-s",
                "GPT_SoVITS/prepare_datasets/"
                "2-get-hubert-wav32k.py",
            ]
            print("[1B]", " ".join(repr(item) for item in cmd))
            ps1b.append(
                Popen(
                    cmd,
                    cwd=now_dir,
                    env=child_env,
                )
            )
    except OSError as exc:
        for process in ps1b:
            if process.poll() is None:
                kill_process(process.pid, process_name_1b)
        ps1b = []
        yield (
            process_info(process_name_1b, "failed") + f": {exc}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    processes = list(ps1b)

    yield (
        process_info(process_name_1b, "running"),
        {"__type__": "update", "visible": False},
        {"__type__": "update", "visible": True},
    )

    return_codes = [process.wait() for process in processes]
    ps1b = []

    if preprocess_1b_cancel_requested:
        preprocess_1b_cancel_requested = False
        return

    if any(code != 0 for code in return_codes):
        yield (
            process_info(process_name_1b, "failed")
            + f" (exit codes: {return_codes})",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    if "Pro" in version:
        try:
            for i_part in range(all_parts):
                child_env = os.environ.copy()
                child_env.update(
                    {
                        **base_config,
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": str(
                            fix_gpu_number(gpu_names[i_part])
                        ),
                    }
                )
                cmd = [
                    python_exec,
                    "-s",
                    "GPT_SoVITS/prepare_datasets/2-get-sv.py",
                ]
                print("[1B-SV]", " ".join(repr(item) for item in cmd))
                ps1b.append(
                    Popen(
                        cmd,
                        cwd=now_dir,
                        env=child_env,
                    )
                )
        except OSError as exc:
            for process in ps1b:
                if process.poll() is None:
                    kill_process(process.pid, process_name_1b)
            ps1b = []
            yield (
                process_info(process_name_1b, "failed") + f": {exc}",
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
            return

        sv_processes = list(ps1b)
        sv_return_codes = [
            process.wait()
            for process in sv_processes
        ]
        ps1b = []

        if preprocess_1b_cancel_requested:
            preprocess_1b_cancel_requested = False
            return

        if any(code != 0 for code in sv_return_codes):
            yield (
                process_info(process_name_1b, "failed")
                + f" (SV exit codes: {sv_return_codes})",
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
            return

    hubert_dir = opt_dir / "4-cnhubert"
    wav32k_dir = opt_dir / "5-wav32k"
    if (
        not hubert_dir.is_dir()
        or not wav32k_dir.is_dir()
        or not any(hubert_dir.iterdir())
        or not any(wav32k_dir.iterdir())
    ):
        yield (
            process_info(process_name_1b, "failed")
            + ": expected feature outputs are missing or empty",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    names_hubert = {
        path.stem
        for path in hubert_dir.iterdir()
        if path.is_file() and path.suffix == ".pt"
    }
    names_wav = {
        path.name
        for path in wav32k_dir.iterdir()
        if path.is_file()
    }
    if names_hubert != names_wav:
        yield (
            process_info(process_name_1b, "failed")
            + ": HuBERT/wav32k sample sets differ",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return
    if len(names_hubert) < 2:
        yield (
            process_info(process_name_1b, "failed")
            + ": fewer than 2 usable audio samples remain",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    write_stage_contract(
        stage_contract_path(opt_dir, "1b"),
        expected_contract,
    )

    yield (
        process_info(process_name_1b, "finish"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


def close1b():
    global ps1b
    global preprocess_1b_cancel_requested

    preprocess_1b_cancel_requested = True

    if ps1b != []:
        for p1b in ps1b:
            try:
                if p1b.poll() is None:
                    kill_process(p1b.pid, process_name_1b)
            except Exception:
                traceback.print_exc()
        ps1b = []
    return (
        process_info(process_name_1b, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps1c = []
preprocess_1c_cancel_requested = False
process_name_1c = i18n("语义Token提取")


def open1c(
    version,
    inp_text,
    inp_wav_dir,
    exp_name,
    gpu_numbers,
    pretrained_s2G_path,
):
    global ps1c
    global preprocess_1c_cancel_requested

    preprocess_1c_cancel_requested = False

    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)

    if not check_for_existance(
        [inp_text, inp_wav_dir],
        is_dataset_processing=True,
    ):
        yield (
            process_info(process_name_1c, "failed"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    check_details(
        [inp_text, inp_wav_dir],
        is_dataset_processing=True,
    )

    pretrained_s2G_path = _clean_user_path(pretrained_s2G_path)
    s2_path = _resolve_repo_path(pretrained_s2G_path)
    if s2_path is None or not s2_path.is_file():
        yield (
            process_info(process_name_1c, "failed")
            + f": S2 checkpoint not found: {s2_path or '<empty>'}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    exp_name = exp_name.rstrip(" ")

    if ps1c != []:
        yield (
            process_info(process_name_1c, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        return

    opt_dir = Path(exp_root) / exp_name
    config_file = (
        "GPT_SoVITS/configs/s2.json"
        if version not in {"v2Pro", "v2ProPlus"}
        else f"GPT_SoVITS/configs/s2{version}.json"
    )
    config_path = _resolve_repo_path(config_file)
    if config_path is None or not config_path.is_file():
        yield (
            process_info(process_name_1c, "failed")
            + f": config not found: {config_path}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    dependency_error = _validate_training_dataset_contract(
        opt_dir,
        inp_text,
        inp_wav_dir,
        version,
        ("1b",),
    )
    if dependency_error:
        yield (
            process_info(process_name_1c, "failed")
            + ": 1B dependency mismatch: "
            + dependency_error,
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    upstream_1b = read_stage_contract(
        stage_contract_path(opt_dir, "1b")
    )
    if upstream_1b is None:
        yield (
            process_info(process_name_1c, "failed")
            + ": 1B provenance disappeared before 1C started",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    expected_contract = _preprocess_contract(
        "1c",
        inp_text,
        inp_wav_dir,
        version,
        {
            "pretrained_s2G": str(s2_path),
            "pretrained_s2G_signature": str(_file_signature(s2_path)),
            "s2config_path": str(config_path),
            "s2config_signature": quick_path_signature(config_path),
            "upstream_1b_digest": str(upstream_1b.get("digest") or ""),
        },
    )
    _ensure_stage_contract(opt_dir, expected_contract)

    base_config = {
        "inp_text": inp_text,
        "exp_name": exp_name,
        "opt_dir": str(opt_dir),
        "pretrained_s2G": str(s2_path),
        "s2config_path": str(config_path),
        "is_half": str(is_half),
        "version": str(version),
    }
    gpu_names = gpu_numbers.split("-")
    all_parts = len(gpu_names)

    try:
        for i_part in range(all_parts):
            child_env = os.environ.copy()
            child_env.update(
                {
                    **base_config,
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": str(
                        fix_gpu_number(gpu_names[i_part])
                    ),
                }
            )
            cmd = [
                python_exec,
                "-s",
                "GPT_SoVITS/prepare_datasets/3-get-semantic.py",
            ]
            print("[1C]", " ".join(repr(item) for item in cmd))
            ps1c.append(
                Popen(
                    cmd,
                    cwd=now_dir,
                    env=child_env,
                )
            )
    except OSError as exc:
        for process in ps1c:
            if process.poll() is None:
                kill_process(process.pid, process_name_1c)
        ps1c = []
        yield (
            process_info(process_name_1c, "failed") + f": {exc}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    processes = list(ps1c)

    yield (
        process_info(process_name_1c, "running"),
        {"__type__": "update", "visible": False},
        {"__type__": "update", "visible": True},
    )

    return_codes = [process.wait() for process in processes]
    ps1c = []

    if preprocess_1c_cancel_requested:
        preprocess_1c_cancel_requested = False
        return

    if any(code != 0 for code in return_codes):
        yield (
            process_info(process_name_1c, "failed")
            + f" (exit codes: {return_codes})",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    opt = ["item_name\tsemantic_audio"]
    path_semantic = opt_dir / "6-name2semantic.tsv"

    try:
        for i_part in range(all_parts):
            semantic_path = (
                opt_dir / f"6-name2semantic-{i_part}.tsv"
            )
            if not semantic_path.is_file():
                raise FileNotFoundError(
                    f"1C output missing: {semantic_path}"
                )
            with semantic_path.open("r", encoding="utf8") as handle:
                opt += handle.read().strip("\n").split("\n")
            semantic_path.unlink()

        path_semantic.parent.mkdir(parents=True, exist_ok=True)
        path_semantic.write_text(
            "\n".join(opt) + "\n",
            encoding="utf8",
        )
    except OSError as exc:
        yield (
            process_info(process_name_1c, "failed") + f": {exc}",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    if len(opt) <= 1:
        yield (
            process_info(process_name_1c, "failed")
            + ": semantic output is empty",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    semantic_names = {
        line.split("\t", 1)[0]
        for line in opt[1:]
        if line.strip()
    }
    hubert_dir = opt_dir / "4-cnhubert"
    hubert_names = {
        path.stem
        for path in hubert_dir.iterdir()
        if path.is_file() and path.suffix == ".pt"
    }
    if semantic_names != hubert_names:
        yield (
            process_info(process_name_1c, "failed")
            + ": semantic/HuBERT sample sets differ",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
        return

    write_stage_contract(
        stage_contract_path(opt_dir, "1c"),
        expected_contract,
    )

    yield (
        process_info(process_name_1c, "finish"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


def close1c():
    global ps1c
    global preprocess_1c_cancel_requested

    preprocess_1c_cancel_requested = True

    if ps1c != []:
        for p1c in ps1c:
            try:
                if p1c.poll() is None:
                    kill_process(p1c.pid, process_name_1c)
            except Exception:
                traceback.print_exc()
        ps1c = []
    return (
        process_info(process_name_1c, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps1abc = []
preprocess_1abc_cancel_requested = False
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
    use_multilingual_bert,
    multilingual_bert_model,
    ssl_pretrained_dir,
    pretrained_s2G_path,
):
    """Run the canonical 1A -> 1B -> 1C implementations in order."""
    global preprocess_1abc_cancel_requested

    preprocess_1abc_cancel_requested = False
    stages = [
        (
            "1A",
            process_name_1a,
            open1a,
            (
                inp_text,
                inp_wav_dir,
                exp_name,
                gpu_numbers1a,
                bert_pretrained_dir,
                use_multilingual_bert,
                multilingual_bert_model,
            ),
        ),
        (
            "1B",
            process_name_1b,
            open1b,
            (
                version,
                inp_text,
                inp_wav_dir,
                exp_name,
                gpu_numbers1Ba,
                ssl_pretrained_dir,
            ),
        ),
        (
            "1C",
            process_name_1c,
            open1c,
            (
                version,
                inp_text,
                inp_wav_dir,
                exp_name,
                gpu_numbers1c,
                pretrained_s2G_path,
            ),
        ),
    ]

    completed = []
    for label, process_name, runner, args in stages:
        if preprocess_1abc_cancel_requested:
            return

        stage_finished = False
        for result in runner(*args):
            if preprocess_1abc_cancel_requested:
                close1abc()
                return

            child_status = result[0] if result else ""
            stage_finished = (
                child_status == process_info(process_name, "finish")
            )
            prefix = ", ".join(
                f"{item}-Done" for item in completed
            )
            progress = f"{label}-Doing"
            if prefix:
                progress = f"{prefix}, {progress}"

            yield (
                i18n("进度") + f": {progress}\n\n{child_status}",
                {"__type__": "update", "visible": False},
                {"__type__": "update", "visible": True},
            )

        if not stage_finished:
            yield (
                process_info(process_name_1abc, "failed")
                + f": {label} did not complete successfully",
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
            return
        completed.append(label)

    yield (
        process_info(process_name_1abc, "finish"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


def close1abc():
    global preprocess_1abc_cancel_requested

    preprocess_1abc_cancel_requested = True

    if ps1a:
        close1a()
    if ps1b:
        close1b()
    if ps1c:
        close1c()

    return (
        process_info(process_name_1abc, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


preprocess_1abcd_cancel_requested = False
process_name_1abcd = "전처리 a → b → c → d"


def open1abcd(
    version,
    inp_text,
    inp_wav_dir,
    exp_name,
    gpu_numbers1a,
    gpu_numbers1Ba,
    gpu_numbers1c,
    bert_pretrained_dir,
    use_multilingual_bert,
    multilingual_bert_model,
    ssl_pretrained_dir,
    pretrained_s2G_path,
    emotion_sidecar,
    emotion_dataset_id,
    emotion_auto_model,
    emotion_auto_confidence,
    emotion_auto_device,
):
    global preprocess_1abcd_cancel_requested

    preprocess_1abcd_cancel_requested = False
    sidecar_update = {"__type__": "update"}

    yield (
        "### 전처리 일괄 실행\n\n**a → b → c 실행 중**",
        {"__type__": "update", "visible": False},
        {"__type__": "update", "visible": True},
        sidecar_update,
    )

    abc_finished = False
    for result in open1abc(
        version,
        inp_text,
        inp_wav_dir,
        exp_name,
        gpu_numbers1a,
        gpu_numbers1Ba,
        gpu_numbers1c,
        bert_pretrained_dir,
        use_multilingual_bert,
        multilingual_bert_model,
        ssl_pretrained_dir,
        pretrained_s2G_path,
    ):
        if preprocess_1abcd_cancel_requested:
            return

        child_status = result[0] if result else ""
        abc_finished = (
            child_status
            == process_info(process_name_1abc, "finish")
        )
        yield (
            "### 전처리 일괄 실행\n\n"
            f"**a → b → c**\n\n{child_status}",
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            sidecar_update,
        )

    if not abc_finished:
        yield (
            "### 전처리 일괄 실행 실패\n\n"
            "a/b/c 중 하나가 완료되지 않아 d 단계는 실행하지 않습니다.",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            sidecar_update,
        )
        return

    if preprocess_1abcd_cancel_requested:
        return

    yield (
        "### 전처리 일괄 실행\n\n"
        "a → b → c 완료. **d 감정 라벨 생성 중**",
        {"__type__": "update", "visible": False},
        {"__type__": "update", "visible": True},
        sidecar_update,
    )

    d_finished = False
    for result in run_auto_emotion_sidecar(
        inp_text,
        inp_wav_dir,
        emotion_sidecar,
        emotion_dataset_id,
        emotion_auto_model,
        emotion_auto_confidence,
        emotion_auto_device,
    ):
        if preprocess_1abcd_cancel_requested:
            return

        child_status = result[0] if result else ""
        child_sidecar = (
            result[1]
            if len(result) > 1
            else {"__type__": "update"}
        )
        if (
            isinstance(child_sidecar, dict)
            and child_sidecar.get("value")
        ):
            sidecar_update = child_sidecar
            d_finished = True

        yield (
            "### 전처리 일괄 실행\n\n"
            f"a → b → c 완료\n\n**d 단계**\n\n{child_status}",
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            sidecar_update,
        )

    if not d_finished:
        yield (
            "### 전처리 일괄 실행 실패\n\n"
            "d 감정 sidecar가 생성되지 않았습니다.",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            sidecar_update,
        )
        return

    yield (
        "### 전처리 일괄 실행 완료\n\n"
        "**a → b → c → d 전부 완료했습니다.**",
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
        sidecar_update,
    )


def close1abcd():
    global preprocess_1abcd_cancel_requested

    preprocess_1abcd_cancel_requested = True

    if ps1abc:
        close1abc()

    if (
        p_emotion_training is not None
        and p_emotion_training.poll() is None
    ):
        stop_emotion_training()

    return (
        "### 전처리 일괄 실행 중지됨",
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
    cmd = [
        python_exec,
        "-s",
        "GPT_SoVITS/download.py",
    ]
    print("[G2PW download]", " ".join(repr(item) for item in cmd))
    result = subprocess.run(
        cmd,
        cwd=now_dir,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "G2PW model download failed with exit code "
            f"{result.returncode}"
        )


def sync(text):
    return {"__type__": "update", "value": text}


with gr.Blocks(title="GPT-SoVITS WebUI", analytics_enabled=False, js=js, css=css, fill_width=True) as app:
    gr.HTML(
        top_html.format(
            i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责.")
            + i18n("如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录LICENSE.")
        ),
        elem_classes="markdown",
    )

    runtime_status_monitor = gr.HTML(
        value=build_runtime_status_html(),
        elem_id="runtime-status-monitor",
        elem_classes="status-monitor-root",
    )
    runtime_status_payload = gr.Textbox(
        value=build_runtime_status_payload(),
        visible=False,
        elem_id="runtime-status-payload",
    )
    runtime_status_timer = gr.Timer(value=2, active=True)
    runtime_status_timer.tick(
        fn=build_runtime_status_payload,
        inputs=[],
        outputs=[runtime_status_payload],
        queue=False,
        show_progress="hidden",
    )
    runtime_status_payload.change(
        fn=None,
        inputs=[runtime_status_payload],
        outputs=[],
        js=runtime_status_patch_js,
        queue=False,
        show_progress="hidden",
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
                close_label.click(close_label_webui, [], [label_info, open_label, close_label])
                open_uvr5.click(change_uvr5, [], [uvr5_info, open_uvr5, close_uvr5])
                close_uvr5.click(close_uvr5_webui, [], [uvr5_info, open_uvr5, close_uvr5])

        with gr.TabItem(i18n("1-GPT-SoVITS-TTS"), id="tts"):
            with gr.Accordion(i18n("微调模型信息")):
                with gr.Column():
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
                with gr.Column():
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
                        with gr.Column():
                            with gr.Row():
                                inp_text = gr.Textbox(
                                    label=i18n("*文本标注文件"),
                                    value="/content/drive/MyDrive/audio_input/training.list",
                                    placeholder="/content/drive/MyDrive/audio_input/training.list",
                                    interactive=True,
                                    scale=10,
                                )
                            with gr.Row():
                                inp_wav_dir = gr.Textbox(
                                    label=i18n("*训练集音频文件目录"),
                                    value="/content/drive/MyDrive/audio_input/datas",
                                    interactive=True,
                                    placeholder="/content/drive/MyDrive/audio_input/datas",
                                    scale=10,
                                )
    
                    with gr.Accordion(label="1Aa-" + process_name_1a):
                        with gr.Column():
                            with gr.Row():
                                gpu_numbers1a = gr.Textbox(
                                    label=i18n("GPU卡号以-分割，每个卡号一个进程"),
                                    value="%s" % (gpus),
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
                                use_multilingual_bert = gr.Checkbox(
                                    label=(
                                        "영어·한국어 multilingual BERT 사용 "
                                        "(새 GPT/S1 파인튜닝용)"
                                    ),
                                    value=True,
                                    interactive=True,
                                )
                                multilingual_bert_model = gr.Textbox(
                                    label="영어·한국어 BERT 모델",
                                    value=(
                                        "google-bert/"
                                        "bert-base-multilingual-cased"
                                    ),
                                    interactive=True,
                                )
                            gr.Markdown(
                                "기존 GPT/S1은 비중국어를 0벡터로 학습했으므로 "
                                "이 옵션은 **새로 GPT/S1을 파인튜닝할 때만** "
                                "사용하세요."
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
                        with gr.Column():
                            with gr.Row():
                                gpu_numbers1Ba = gr.Textbox(
                                    label=i18n("GPU卡号以-分割，每个卡号一个进程"),
                                    value="%s" % (gpus),
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
                        with gr.Column():
                            with gr.Row():
                                gpu_numbers1c = gr.Textbox(
                                    label=i18n("GPU卡号以-分割，每个卡号一个进程"),
                                    value="%s" % (gpus),
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

자동 생성은 emotion2vec+의 절대 top-1을 그대로 쓰지 않고,
**각 화자의 평소 음성 분포를 기준으로 상대적인 감정 편차**를 계산합니다.
화자 기준에서 뚜렷한 편차가 없는 샘플은 neutral/default로 처리하고,
controller는 비-neutral 방향만 학습합니다. 동일 전사문 pair는 필요하지 않습니다.
자동 결과가 마음에 들지 않으면 sidecar JSONL을 직접 수정할 수 있습니다.

경로를 비워두면 감정 관련 산출물은 다른 전처리 도구와 마찬가지로
`output/emotion_opt/<데이터셋 ID>/` 아래에 정리됩니다.
"""
                        )

                        with gr.Row():
                            emotion_auto_model = gr.Dropdown(
                                label="감정 분석 모델 (ASR과 별개)",
                                choices=[
                                    (
                                        "emotion2vec+ large (기본, 정확도 우선)",
                                        "iic/emotion2vec_plus_large",
                                    ),
                                    (
                                        "emotion2vec+ base (가벼운 대안)",
                                        "iic/emotion2vec_plus_base",
                                    ),
                                ],
                                value="iic/emotion2vec_plus_large",
                                interactive=True,
                            )
                            emotion_auto_confidence = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                step=0.05,
                                label="자동 감정 라벨 최소 신뢰도",
                                value=0.35,
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
                                    "비워두면 output/emotion_opt/"
                                    "<데이터셋 ID>/emotion_sidecar.jsonl"
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
                                value="음성에서 상대 감정 라벨 생성",
                                variant="primary",
                            )
                            emotion_auto_stop = gr.Button(
                                value="자동 감정 분석 중지",
                                variant="stop",
                            )

                        emotion_auto_status = gr.Markdown(
                            value=(
                                "emotion2vec+ 감정 분석 → 화자 기준 보정 → "
                                "상대 편차 필터 → neutral/default 분리 → sidecar 생성"
                            )
                        )

                        with gr.Row():
                            emotion_dataset_dir = gr.Textbox(
                                label="감정 canonical dataset 출력 폴더",
                                value="",
                                placeholder=(
                                    "비워두면 output/emotion_opt/"
                                    "<데이터셋 ID>/dataset"
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
                                emotion_dataset_id,
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

                    with gr.Accordion(label="1Aabcd-" + process_name_1abcd):
                        with gr.Column():
                            with gr.Row():
                                button1abc_open = gr.Button(
                                    value="a → b → c → d 순서대로 실행", variant="primary", visible=True
                                )
                                button1abc_close = gr.Button(
                                    value="일괄 전처리 중지", variant="stop", visible=False
                                )
                            with gr.Row():
                                info1abc = gr.Markdown(value="위 a/b/c/d 설정을 확인한 뒤 실행하세요.")
    

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
                    [
                        inp_text,
                        inp_wav_dir,
                        exp_name,
                        gpu_numbers1a,
                        bert_pretrained_dir,
                        use_multilingual_bert,
                        multilingual_bert_model,
                    ],
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
                    open1abcd,
                    [
                        version_checkbox,
                        inp_text,
                        inp_wav_dir,
                        exp_name,
                        gpu_numbers1a,
                        gpu_numbers1Ba,
                        gpu_numbers1c,
                        bert_pretrained_dir,
                        use_multilingual_bert,
                        multilingual_bert_model,
                        cnhubert_base_dir,
                        pretrained_s2G,
                        emotion_sidecar,
                        emotion_dataset_id,
                        emotion_auto_model,
                        emotion_auto_confidence,
                        emotion_auto_device,
                    ],
                    [
                        info1abc,
                        button1abc_open,
                        button1abc_close,
                        emotion_sidecar,
                    ],
                )
                button1abc_close.click(
                    close1abcd,
                    [],
                    [
                        info1abc,
                        button1abc_open,
                        button1abc_close,
                    ],
                    queue=False,
                )
    
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
                        with gr.Column():
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
                                        label=(i18n("是否开启DPO训练选项(实验性)") + " · 기본 ON, 실제 batch 자동 1/2"),
                                        value=True,
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
                        with gr.Column():
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
                        label="1Bc-" + i18n("GPT → SoVITS 연속 학습")
                    ):
                        with gr.Column():
                            with gr.Row():
                                button_train_sequence_open = gr.Button(
                                    value=process_info(
                                        process_name_train_sequence,
                                        "open",
                                    ),
                                    variant="primary",
                                    visible=True,
                                )
                                button_train_sequence_close = gr.Button(
                                    value=process_info(
                                        process_name_train_sequence,
                                        "close",
                                    ),
                                    variant="primary",
                                    visible=False,
                                )
                            with gr.Row():
                                info_train_sequence = gr.Textbox(
                                    label=process_info(
                                        process_name_train_sequence,
                                        "info",
                                    )
                                )

                    with gr.Accordion(
                        label="1Bd-감정 제어 학습 (V5dev, 선택)",
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
                                    "비워두면 output/emotion_opt/"
                                    "<데이터셋 ID>/features"
                                ),
                                interactive=True,
                            )
                            emotion_controller_output = gr.Textbox(
                                label="Controller 저장 경로",
                                value="",
                                placeholder=(
                                    "비워두면 output/emotion_opt/<데이터셋 ID>/"
                                    "<데이터셋 ID>_emotion_controller.pt"
                                ),
                                interactive=True,
                            )

                        with gr.Row():
                            emotion_s1_checkpoint = gr.Textbox(
                                label=(
                                    "Controller에 사용할 GPT/S1 체크포인트 "
                                    "(비우면 현재 실험의 최신 GPT 자동 선택)"
                                ),
                                value="",
                                placeholder=(
                                    "비우면 GPT_weights_v5dev/<현재 실험명>-e*.ckpt "
                                    "중 최신 epoch 자동 선택"
                                ),
                                interactive=True,
                            )

                        with gr.Row():
                            emotion_names = gr.Textbox(
                                label="감정 목록 (선택)",
                                value="",
                                placeholder=(
                                    "비워두면 실제 비-neutral 라벨에서 자동 감지 "
                                    "(예: happy,sad,angry,surprised)"
                                ),
                                interactive=True,
                            )
                            emotion_steps = gr.Number(
                                label="기본 학습 steps (val 기반 자동 연장)",
                                value=500,
                                minimum=100,
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
                            exp_name,
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
                    with gr.Column():
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
                            close_tts_inference,
                            [],
                            [tts_info, open_tts, close_tts],
                        )
                button_train_sequence_open.click(
                    run_gpt_then_sovits,
                    [
                        batch_size1Bb,
                        total_epoch1Bb,
                        exp_name,
                        inp_text,
                        inp_wav_dir,
                        if_dpo,
                        if_save_latest1Bb,
                        if_save_every_weights1Bb,
                        save_every_epoch1Bb,
                        gpu_numbers1Bb,
                        pretrained_s1,
                        use_multilingual_bert,
                        multilingual_bert_model,
                        version_checkbox,
                        batch_size,
                        total_epoch,
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
                    [
                        info_train_sequence,
                        button_train_sequence_open,
                        button_train_sequence_close,
                        SoVITS_dropdown,
                        GPT_dropdown,
                    ],
                )
                button_train_sequence_close.click(
                    stop_gpt_then_sovits,
                    [],
                    [
                        info_train_sequence,
                        button_train_sequence_open,
                        button_train_sequence_close,
                    ],
                    queue=False,
                )

                button1Ba_open.click(
                    open1Ba,
                    [
                        version_checkbox,
                        batch_size,
                        total_epoch,
                        exp_name,
                        inp_text,
                        inp_wav_dir,
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
                        inp_text,
                        inp_wav_dir,
                        if_dpo,
                        if_save_latest1Bb,
                        if_save_every_weights1Bb,
                        save_every_epoch1Bb,
                        gpu_numbers1Bb,
                        pretrained_s1,
                        pretrained_s2G,
                        use_multilingual_bert,
                        multilingual_bert_model,
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
        share=True,
        server_port=webui_port_main,
        # quiet=True,
    )
