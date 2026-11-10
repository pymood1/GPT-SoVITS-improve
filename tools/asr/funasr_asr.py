# -*- coding:utf-8 -*-

import argparse
import os
import traceback

from funasr import AutoModel
from modelscope import snapshot_download
from tqdm import tqdm

funasr_models = {}

FUN_ASR_NANO_MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512"
FUN_ASR_MLT_NANO_MODEL_ID = "FunAudioLLM/Fun-ASR-MLT-Nano-2512"

FUN_ASR_NANO_MODEL_SOURCES = (
    {
        "model": FUN_ASR_NANO_MODEL_ID,
        "hub": "hf",
        "trust_remote_code": True,
    },
    {
        "model": FUN_ASR_NANO_MODEL_ID,
        "hub": "ms",
        "trust_remote_code": False,
    },
)

FUN_ASR_MLT_NANO_MODEL_SOURCES = (
    {
        "model": FUN_ASR_MLT_NANO_MODEL_ID,
        "hub": "hf",
        "trust_remote_code": True,
    },
    {
        "model": FUN_ASR_MLT_NANO_MODEL_ID,
        "hub": "ms",
        "trust_remote_code": False,
    },
)


FUN_ASR_LANGUAGE_NAMES = {
    "zh": "中文",
    "en": "英文",
    "yue": "粤语",
    "ja": "日文",
    "ko": "韩文",
}


def only_asr(input_file, language, backend="fun-asr-nano"):
    try:
        model = create_model(language, backend=backend)
        generate_kwargs = {"input": input_file}

        if backend in {"fun-asr-nano", "fun-asr-mlt-nano"}:
            generate_kwargs["language"] = FUN_ASR_LANGUAGE_NAMES.get(
                language,
                language,
            )
            generate_kwargs["itn"] = True
        elif backend == "sensevoice":
            generate_kwargs["language"] = language
            generate_kwargs["use_itn"] = True

        text = model.generate(**generate_kwargs)[0]["text"]
    except Exception:
        text = ""
        print(traceback.format_exc())
    return text


def is_unregistered_model_error(exc):
    message = str(exc).lower()
    return "model" in message and "not registered" in message


def create_fun_asr_model(model_sources, device):
    common_kwargs = {
        "vad_model": "fsmn-vad",
        "device": device,
        "disable_update": True,
    }
    last_error = None

    for model_source in model_sources:
        try:
            return AutoModel(
                **model_source,
                **common_kwargs,
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            if not is_unregistered_model_error(exc):
                raise
            last_error = exc

    if last_error is not None:
        raise last_error

    raise RuntimeError("No Fun-ASR model source is configured")


def create_model(language="zh", **kwargs):
    backend = kwargs.get("backend", "fun-asr-nano")

    if backend in {
        "fun-asr-nano",
        "fun-asr-mlt-nano",
        "sensevoice",
    }:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_key = f"{backend}:{device}"

        if cache_key in funasr_models:
            return funasr_models[cache_key]

        if backend == "fun-asr-nano":
            model = create_fun_asr_model(
                FUN_ASR_NANO_MODEL_SOURCES,
                device,
            )
            model_name = "Fun-ASR-Nano"
        elif backend == "fun-asr-mlt-nano":
            model = create_fun_asr_model(
                FUN_ASR_MLT_NANO_MODEL_SOURCES,
                device,
            )
            model_name = "Fun-ASR-MLT-Nano"
        else:
            model = AutoModel(
                model="iic/SenseVoiceSmall",
                vad_model="fsmn-vad",
                device=device,
                disable_update=True,
            )
            model_name = "SenseVoice"

        print(
            f"FunASR {model_name} model loaded: "
            f"{language.upper()} / {device}"
        )
        funasr_models[cache_key] = model
        return model

    if backend != "classic":
        raise ValueError(f"Unknown FunASR backend: {backend}")

    if language == "zh":
        path_vad = (
            "tools/asr/models/"
            "speech_fsmn_vad_zh-cn-16k-common-pytorch"
        )
        path_punc = (
            "tools/asr/models/"
            "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
        )
        path_asr = (
            "tools/asr/models/"
            "speech_paraformer-large_asr_nat-zh-cn-16k-"
            "common-vocab8404-pytorch"
        )

        snapshot_download(
            "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            local_dir=path_vad,
        )
        snapshot_download(
            "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            local_dir=path_punc,
        )
        snapshot_download(
            "iic/speech_paraformer-large_asr_nat-zh-cn-16k-"
            "common-vocab8404-pytorch",
            local_dir=path_asr,
        )

        model_revision = "v2.0.4"
        vad_model_revision = "v2.0.4"
        punc_model_revision = "v2.0.4"

    elif language == "yue":
        path_asr = (
            "tools/asr/models/"
            "speech_UniASR_asr_2pass-cantonese-CHS-16k-"
            "common-vocab1468-tensorflow1-online"
        )

        snapshot_download(
            "iic/speech_UniASR_asr_2pass-cantonese-CHS-16k-"
            "common-vocab1468-tensorflow1-online",
            local_dir=path_asr,
        )

        path_vad = None
        path_punc = None
        vad_model_revision = ""
        punc_model_revision = ""
        model_revision = "master"

    else:
        raise ValueError(
            "Classic FunASR supports only zh/yue in this UI"
        )

    cache_key = f"classic:{language}"

    if cache_key in funasr_models:
        return funasr_models[cache_key]

    model = AutoModel(
        model=path_asr,
        model_revision=model_revision,
        vad_model=path_vad,
        vad_model_revision=vad_model_revision,
        punc_model=path_punc,
        punc_model_revision=punc_model_revision,
    )
    print(f"FunASR classic model loaded: {language.upper()}")

    funasr_models[cache_key] = model
    return model


def execute_asr(
    input_folder,
    output_folder,
    model_size,
    language,
    backend="fun-asr-nano",
):
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

    input_file_names = sorted(os.listdir(input_folder))
    output = []

    model = create_model(
        language,
        backend=backend,
    )

    for file_name in tqdm(input_file_names):
        try:
            print("\n" + file_name)
            file_path = os.path.join(input_folder, file_name)

            if not os.path.isfile(file_path):
                continue

            generate_kwargs = {"input": file_path}

            if backend in {"fun-asr-nano", "fun-asr-mlt-nano"}:
                generate_kwargs["language"] = FUN_ASR_LANGUAGE_NAMES.get(
                    language,
                    language,
                )
                generate_kwargs["itn"] = True
            elif backend == "sensevoice":
                generate_kwargs["language"] = language
                generate_kwargs["use_itn"] = True

            result = model.generate(**generate_kwargs)
            text = result[0]["text"]

            if backend == "sensevoice":
                from funasr.utils.postprocess_utils import (
                    rich_transcription_postprocess,
                )

                text = rich_transcription_postprocess(text)

            text = str(text or "").strip()
            if not text:
                print(f"Empty ASR transcript: {file_path}")
                continue

            output.append(
                f"{file_path}|{output_file_name}|"
                f"{language.upper()}|{text}"
            )
        except Exception:
            print(traceback.format_exc())

    if not output:
        raise RuntimeError(
            "FunASR produced no usable GPT-SoVITS transcripts"
        )

    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.abspath(
        f"{output_folder}/{output_file_name}.list"
    )

    with open(output_file_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(output))
        print(
            f"ASR task complete -> {output_file_path}\n"
        )

    return output_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        required=True,
        help="Path to the folder containing audio files",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        required=True,
        help="Output folder for GPT-SoVITS .list files",
    )
    parser.add_argument(
        "-s",
        "--model_size",
        type=str,
        default="large",
        help="Compatibility option for the shared ASR UI",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="zh",
        choices=["zh", "yue", "ja", "en", "ko"],
        help="GPT-SoVITS language label",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="float32",
        choices=["float16", "float32"],
        help="Compatibility option; FunASR selects its own runtime dtype",
    )
    parser.add_argument(
        "--backend",
        choices=[
            "fun-asr-nano",
            "fun-asr-mlt-nano",
            "sensevoice",
            "classic",
        ],
        default="fun-asr-nano",
    )

    cmd = parser.parse_args()

    execute_asr(
        input_folder=cmd.input_folder,
        output_folder=cmd.output_folder,
        model_size=cmd.model_size,
        language=cmd.language,
        backend=cmd.backend,
    )
