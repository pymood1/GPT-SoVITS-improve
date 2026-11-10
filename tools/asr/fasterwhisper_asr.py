import argparse
import os
import traceback

import requests
import torch
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download as snapshot_download_hf
from modelscope import snapshot_download as snapshot_download_ms
from tqdm import tqdm

from tools.asr.config import get_models
from tools.my_utils import load_cudnn

# fmt: off
GPT_SOVITS_LANGUAGES = {"zh", "en", "ja", "ko"}

language_code_list = [
    "af", "am", "ar", "as", "az", 
    "ba", "be", "bg", "bn", "bo", 
    "br", "bs", "ca", "cs", "cy", 
    "da", "de", "el", "en", "es", 
    "et", "eu", "fa", "fi", "fo", 
    "fr", "gl", "gu", "ha", "haw", 
    "he", "hi", "hr", "ht", "hu", 
    "hy", "id", "is", "it", "ja", 
    "jw", "ka", "kk", "km", "kn", 
    "ko", "la", "lb", "ln", "lo", 
    "lt", "lv", "mg", "mi", "mk", 
    "ml", "mn", "mr", "ms", "mt", 
    "my", "ne", "nl", "nn", "no", 
    "oc", "pa", "pl", "ps", "pt", 
    "ro", "ru", "sa", "sd", "si", 
    "sk", "sl", "sn", "so", "sq", 
    "sr", "su", "sv", "sw", "ta", 
    "te", "tg", "th", "tk", "tl", 
    "tr", "tt", "uk", "ur", "uz", 
    "vi", "yi", "yo", "zh",
    "auto"] 
# fmt: on


def download_model(model_size: str):
    url = "https://huggingface.co/api/models/gpt2"
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        source = "HF"
    except Exception:
        source = "ModelScope"

    model_path = ""
    if source == "HF":
        if "distil" in model_size:
            if "3.5" in model_size:
                repo_id = "distil-whisper/distil-large-v3.5-ct2"
                model_path = "tools/asr/models/faster-distil-whisper-large-v3.5"
            else:
                repo_id = "Systran/faster-{}-whisper-{}".format(*model_size.split("-", maxsplit=1))
        elif model_size == "large-v3-turbo":
            repo_id = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
            model_path = "tools/asr/models/faster-whisper-large-v3-turbo"
        else:
            repo_id = f"Systran/faster-whisper-{model_size}"
        model_path = (
            model_path or f"tools/asr/models/{repo_id.replace('Systran/', '').replace('distil-whisper/', '', 1)}"
        )
    else:
        repo_id = "XXXXRT/faster-whisper"
        model_path = "tools/asr/models"

    files: list[str] = [
        "config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.txt",
    ]
    if "large-v3" in model_size or "distil" in model_size:
        files.append("preprocessor_config.json")
        files.append("vocabulary.json")

        files.remove("vocabulary.txt")

    if source == "HF":
        print(
            f"Downloading model from HuggingFace: "
            f"{repo_id} to {model_path}"
        )
        try:
            snapshot_download_hf(
                repo_id,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                allow_patterns=files,
            )
            return model_path
        except Exception as exc:
            print(
                "[Faster Whisper] HuggingFace download failed; "
                f"falling back to ModelScope: {exc}",
                flush=True,
            )

    repo_id = "XXXXRT/faster-whisper"
    model_path = "tools/asr/models"
    ms_files = [
        (
            f"faster-whisper-{model_size}/{file}"
            .replace("whisper-distil", "distil-whisper")
        )
        for file in files
    ]
    print(
        f"Downloading model from ModelScope: "
        f"{repo_id} to {model_path}"
    )
    snapshot_download_ms(
        repo_id,
        local_dir=model_path,
        allow_patterns=ms_files,
    )
    return (
        model_path
        + f"/faster-whisper-{model_size}".replace(
            "whisper-distil",
            "distil-whisper",
        )
    )


def execute_asr(input_folder, output_folder, model_path, language, precision):
    input_folder = os.path.abspath(
        os.path.expanduser(str(input_folder))
    )
    if not os.path.isdir(input_folder):
        raise RuntimeError(
            f"ASR input directory not found: {input_folder}"
        )

    output_folder = os.path.abspath(
        os.path.expanduser(output_folder or "output/asr_opt")
    )
    output_file_name = os.path.basename(
        os.path.normpath(input_folder)
    )
    if not output_file_name:
        raise RuntimeError(
            f"ASR input directory has no usable name: {input_folder}"
        )
    if "|" in input_folder or "|" in output_file_name:
        raise RuntimeError(
            "GPT-SoVITS .list paths/speaker names must not contain '|'"
        )

    if language == "auto":
        language = None  # 不设置语种由模型自动输出概率最高的语种
    print("loading faster whisper model:", model_path, model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(model_path, device=device, compute_type=precision)

    input_file_names = os.listdir(input_folder)
    input_file_names.sort()

    output = []

    for file_name in tqdm(input_file_names):
        try:
            file_path = os.path.join(input_folder, file_name)

            if "|" in file_path:
                print(
                    "Skipping path that cannot be represented in "
                    f"GPT-SoVITS .list: {file_path}"
                )
                continue

            segments, info = model.transcribe(
                audio=file_path,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=700),
                language=language,
            )
            detected_language = str(
                info.language or language or ""
            ).lower()

            if detected_language not in GPT_SOVITS_LANGUAGES:
                print(
                    "Unsupported GPT-SoVITS language detected by Whisper: "
                    f"{detected_language or 'unknown'} ({file_path})"
                )
                continue

            text = "".join(
                segment.text for segment in segments
            )
            text = " ".join(str(text).splitlines()).replace(
                "|",
                "｜",
            ).strip()

            if not text:
                print(f"Empty Whisper transcript: {file_path}")
                continue

            output.append(
                f"{file_path}|{output_file_name}|"
                f"{detected_language.upper()}|{text}"
            )
        except Exception as e:
            print(e)
            traceback.print_exc()

    if not output:
        raise RuntimeError(
            "Faster Whisper produced no usable GPT-SoVITS transcripts"
        )

    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.abspath(f"{output_folder}/{output_file_name}.list")

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))
        print(f"ASR 任务完成->标注文件路径: {output_file_path}\n")
    return output_file_path


load_cudnn()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_folder", type=str, required=True, help="Path to the folder containing WAV files."
    )
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Output folder to store transcriptions.")
    parser.add_argument(
        "-s",
        "--model_size",
        type=str,
        default="large-v3",
        choices=get_models(),
        help="Model Size of Faster Whisper",
    )
    parser.add_argument(
        "-l", "--language", type=str, default="ja", choices=language_code_list, help="Language of the audio files."
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="float16",
        choices=["float16", "float32", "int8"],
        help="fp16, int8 or fp32",
    )

    cmd = parser.parse_args()
    model_size = cmd.model_size
    if model_size == "large":
        model_size = "large-v3"
    model_path = download_model(model_size)
    output_file_path = execute_asr(
        input_folder=cmd.input_folder,
        output_folder=cmd.output_folder,
        model_path=model_path,
        language=cmd.language,
        precision=cmd.precision,
    )
