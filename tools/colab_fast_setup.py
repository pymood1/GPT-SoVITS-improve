from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil
import subprocess
import sys
import time

T0 = time.perf_counter()
REPO_DIR = Path(__file__).resolve().parents[1]
DL_DIR = Path("/content/gpt_sovits_downloads")
DL_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_UVR5 = False
HF_HOME = Path("/content/huggingface")
HF_HOME.mkdir(parents=True, exist_ok=True)
EMOTION_MODEL_DIRS = {
    "large": (
        REPO_DIR / "runtime" / "models" / "emotion2vec_plus_large"
    ),
    "base": (
        REPO_DIR / "runtime" / "models" / "emotion2vec_plus_base"
    ),
}

HF_ASSETS = {
    "pretrained_models.zip": (
        "XXXXRT/GPT-SoVITS-Pretrained",
        "pretrained_models.zip",
    ),
    "G2PWModel.zip": (
        "XXXXRT/GPT-SoVITS-Pretrained",
        "G2PWModel.zip",
    ),
    "nltk_data.zip": (
        "XXXXRT/GPT-SoVITS-Pretrained",
        "nltk_data.zip",
    ),
    "open_jtalk_dic_utf_8-1.11.tar.gz": (
        "XXXXRT/GPT-SoVITS-Pretrained",
        "open_jtalk_dic_utf_8-1.11.tar.gz",
    ),
    "s2Gv5dev.pth": (
        "lj1995/GPT-SoVITS",
        "gsv-v5-pretrained/s2Gv5dev.pth",
    ),
    "v5-vocoder.pth": (
        "lj1995/GPT-SoVITS",
        "gsv-v5-pretrained/vocoder.pth",
    ),
}
if DOWNLOAD_UVR5:
    HF_ASSETS["uvr5_weights.zip"] = (
        "XXXXRT/GPT-SoVITS-Pretrained",
        "uvr5_weights.zip",
    )


def elapsed(label, start):
    print(f"[time] {label}: {time.perf_counter() - start:.1f}s")
    return time.perf_counter()


def run(cmd, *, cwd=None, env=None, check=True, quiet=False, capture=False):
    cmd = [str(x) for x in cmd]
    if not quiet:
        print("+", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=check,
        text=True,
        capture_output=capture,
    )


# 0. Fast runtime check.
# Do NOT import torch/torchaudio here: on a fresh Colab runtime their first import
# can take ~45-60 seconds and makes the first cell look frozen.
print("[setup] fast runtime preflight...", flush=True)
if not ((3, 10) <= sys.version_info[:2] <= (3, 14)):
    raise RuntimeError(
        "Supported Python versions are 3.10 through 3.14."
    )

try:
    from importlib.metadata import version as pkg_version
    print("Python:", sys.version.split()[0], flush=True)
    print("PyTorch package:", pkg_version("torch"), flush=True)
    print("torchaudio package:", pkg_version("torchaudio"), flush=True)
except Exception as exc:
    print("[warn] package metadata check:", repr(exc), flush=True)

if shutil.which("nvidia-smi"):
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True, capture_output=True, check=False,
    )
    print("GPU:", gpu.stdout.strip() or "nvidia-smi available", flush=True)
else:
    raise RuntimeError("GPU runtime is required.")

t = elapsed("runtime preflight", T0)


# 1. Reuse system tools already present in Colab.
required = ["ffmpeg", "cmake", "make", "unzip", "gcc", "g++", "curl", "tar"]
missing = [x for x in required if shutil.which(x) is None]
if missing:
    mapping = {
        "ffmpeg": "ffmpeg", "cmake": "cmake", "make": "make",
        "unzip": "unzip", "gcc": "gcc", "g++": "g++",
        "curl": "curl", "tar": "tar",
    }
    run(["apt-get", "update", "-qq"])
    run(["apt-get", "install", "-y", "-qq", *sorted({mapping[x] for x in missing})])
else:
    print("System tools already present; apt/conda install skipped.", flush=True)
t = elapsed("system tools", t)


# 2. uv bootstrap.
if shutil.which("uv") is None:
    run([sys.executable, "-m", "pip", "install", "-q", "uv"])
UV = shutil.which("uv")
if not UV:
    raise RuntimeError("uv installation failed.")

uv_env = os.environ.copy()
uv_env.update({
    "UV_CONCURRENT_DOWNLOADS": "16",
    "UV_CONCURRENT_BUILDS": "4",
    "UV_CONCURRENT_INSTALLS": "8",
    "UV_HTTP_TIMEOUT": "120",
    "HF_HOME": str(HF_HOME),
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_UPDATE_CHECK": "1",
    "HF_HUB_DOWNLOAD_TIMEOUT": "120",
    "HF_XET_HIGH_PERFORMANCE": "1",
})
os.environ.update({
    key: value
    for key, value in uv_env.items()
    if key.startswith("HF_")
})

run([UV, "--version"], env=uv_env)
print("[setup] enabling Hugging Face Xet transport...", flush=True)
run(
    [
        UV,
        "pip",
        "install",
        "--system",
        "--quiet",
        "huggingface_hub>=0.32,<1",
        "hf_xet>=1.7",
    ],
    env=uv_env,
)
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()

t = elapsed("uv + HF Xet bootstrap", t)


# 3. Keep the COMPLETE repository requirements.
#    Python-version markers in requirements.txt are the single source of truth
#    for NumPy, G2P backends, Pydantic, and other compatibility-sensitive deps.
repo_req = REPO_DIR / "requirements.txt"
repo_extra = REPO_DIR / "extra-req.txt"
lines = repo_req.read_text(encoding="utf-8").splitlines()

full_req = Path("/content/gpt_sovits_full_requirements.txt")
full_req.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("[setup] preparing full requirements...", flush=True)
print("Full requirements retained:", sum(
    1 for x in lines if x.strip() and not x.lstrip().startswith("#")
))
print("extra-req.txt retained:", repo_extra.read_text(encoding="utf-8").strip())
if sys.version_info >= (3, 13):
    print("NumPy ABI target: 2.x (modern Python path)")
else:
    print("NumPy ABI target: 1.26.x-compatible path")
print("No GPT-SoVITS dependency is intentionally excluded.")


# 4. Download installer/model assets while uv installs Python deps.
#    HF progress bars are disabled because Colab stores every redraw in the
#    cell output and eventually becomes sluggish. We print only phase changes.
def _fallback_curl(name, repo_id, filename):
    dest = DL_DIR / name
    part = DL_DIR / f"{name}.part"
    url = (
        f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    )
    cmd = [
        "curl", "-L", "--fail", "--silent", "--show-error",
        "--retry", "8", "--retry-delay", "2",
        "--connect-timeout", "20",
        "--speed-time", "30", "--speed-limit", "1024",
        "-C", "-", "-o", str(part), url,
    ]
    subprocess.run(cmd, check=True)
    part.replace(dest)
    return dest


def download_one(name, repo_id, filename):
    dest = DL_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        print(
            f"[download] reuse {name} "
            f"({dest.stat().st_size / 1024**2:.1f} MiB)",
            flush=True,
        )
        return dest

    print(f"[download] start {name} (HF Xet)", flush=True)
    try:
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(DL_DIR),
                cache_dir=str(HF_HOME / "hub"),
            )
        )
        if downloaded.resolve() != dest.resolve():
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            try:
                dest.symlink_to(downloaded)
            except OSError:
                shutil.copy2(downloaded, dest)
    except Exception as exc:
        print(
            f"[download] Xet fallback for {name}: "
            f"{type(exc).__name__}",
            flush=True,
        )
        _fallback_curl(name, repo_id, filename)

    print(
        f"[download] done  {name} "
        f"({dest.stat().st_size / 1024**2:.1f} MiB)",
        flush=True,
    )
    return dest


def download_emotion_models():
    specs = (
        (
            "large",
            "emotion2vec/emotion2vec_plus_large",
            EMOTION_MODEL_DIRS["large"],
        ),
        (
            "base",
            "emotion2vec/emotion2vec_plus_base",
            EMOTION_MODEL_DIRS["base"],
        ),
    )
    completed = []

    # Download the large model first because it is the default classifier.
    # Base is kept locally as a fallback/comparison model.
    for label, repo_id, model_dir in specs:
        marker = model_dir / "model.pt"
        if marker.is_file():
            print(
                f"[download] reuse emotion2vec+ {label} "
                f"({marker.stat().st_size / 1024**3:.2f} GiB weight)",
                flush=True,
            )
            completed.append(model_dir)
            continue

        model_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[download] start emotion2vec+ {label} "
            "(HF Xet, progress hidden)",
            flush=True,
        )
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
            cache_dir=str(HF_HOME / "hub"),
        )
        if not marker.is_file():
            raise FileNotFoundError(
                f"emotion2vec {label} weight missing after download: {marker}"
            )
        print(
            f"[download] done  emotion2vec+ {label} "
            f"({marker.stat().st_size / 1024**3:.2f} GiB weight)",
            flush=True,
        )
        completed.append(model_dir)

    return completed


pip_cmd = [
    UV, "pip", "install", "--system", "--quiet",
    "-r", str(full_req),
    "torchcodec",
]
pip_log_path = Path("/content/gpt_sovits_dependency_install.log")
print(
    "[setup] dependencies + model downloads running in parallel...",
    flush=True,
)
print(
    "[setup] detailed dependency output -> "
    f"{pip_log_path}",
    flush=True,
)

with pip_log_path.open("w", encoding="utf-8") as pip_log:
    pip_proc = subprocess.Popen(
        pip_cmd,
        cwd=str(REPO_DIR),
        env=uv_env,
        text=True,
        stdout=pip_log,
        stderr=subprocess.STDOUT,
    )

    errors = []
    emotion_error = None
    jobs = [
        (name, repo_id, filename)
        for name, (repo_id, filename) in HF_ASSETS.items()
    ]

    # Xet already performs internal adaptive concurrency. Keep outer
    # concurrency low so multiple large transfers do not fight each other.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(download_one, name, repo_id, filename): name
            for name, repo_id, filename in jobs
        }
        emotion_future = pool.submit(download_emotion_models)

        for future in as_completed(list(futures) + [emotion_future]):
            if future is emotion_future:
                try:
                    future.result()
                except Exception as exc:
                    emotion_error = repr(exc)
                continue

            name = futures[future]
            try:
                future.result()
            except Exception as exc:
                errors.append((name, repr(exc)))

    rc = pip_proc.wait()

if rc:
    tail = pip_log_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()[-80:]
    print("\n".join(tail), file=sys.stderr)
    raise subprocess.CalledProcessError(rc, pip_cmd)

print("[deps] full Python dependency install complete", flush=True)

# Same behavior as upstream install.sh for faster-whisper, but keep normal
# successful output out of the Colab cell.
extra = subprocess.run(
    [
        UV, "pip", "install", "--system", "--quiet",
        "-r", str(repo_extra), "--no-deps",
    ],
    cwd=str(REPO_DIR),
    env=uv_env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
if extra.returncode:
    print(extra.stdout, file=sys.stderr)
    raise subprocess.CalledProcessError(extra.returncode, extra.args)

if errors:
    raise RuntimeError(
        "Asset download failed:\n"
        + "\n".join(f"{n}: {e}" for n, e in errors)
    )
if emotion_error:
    print(
        "[warn] emotion2vec prefetch failed; automatic labeling may "
        "download it later:\n" + emotion_error,
        file=sys.stderr,
        flush=True,
    )

t = elapsed("full dependencies + Xet downloads", t)


# 5. Extract independent archives in parallel.
def extract_pretrained():
    marker = REPO_DIR / "GPT_SoVITS/pretrained_models/sv"
    if marker.exists():
        return "[extract] pretrained already present"
    run(["unzip", "-q", "-o", str(DL_DIR / "pretrained_models.zip"),
         "-d", str(REPO_DIR / "GPT_SoVITS")], quiet=True)
    return "[extract] pretrained_models.zip done"


def extract_g2pw():
    marker = REPO_DIR / "GPT_SoVITS/text/G2PWModel"
    if marker.exists():
        return "[extract] G2PW already present"
    run(["unzip", "-q", "-o", str(DL_DIR / "G2PWModel.zip"),
         "-d", str(REPO_DIR / "GPT_SoVITS/text")], quiet=True)
    return "[extract] G2PWModel.zip done"


def extract_nltk():
    runtime_dir = REPO_DIR / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    marker = runtime_dir / "nltk_data"

    if marker.exists():
        return "[extract] NLTK already present"

    run(
        [
            "unzip",
            "-q",
            "-o",
            str(DL_DIR / "nltk_data.zip"),
            "-d",
            str(runtime_dir),
        ],
        quiet=True,
    )

    return "[extract] nltk_data.zip done"


def extract_openjtalk():
    code = "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))"
    proc = run([sys.executable, "-c", code], capture=True, quiet=True)
    pkg = Path(proc.stdout.strip())
    marker = pkg / "open_jtalk_dic_utf_8-1.11"
    if marker.exists():
        return "[extract] OpenJTalk already present"
    run(["tar", "-xzf", str(DL_DIR / "open_jtalk_dic_utf_8-1.11.tar.gz"),
         "-C", str(pkg)], quiet=True)
    return "[extract] OpenJTalk dictionary done"


with ThreadPoolExecutor(max_workers=4) as pool:
    fs = [pool.submit(fn) for fn in (
        extract_pretrained, extract_g2pw, extract_nltk, extract_openjtalk
    )]
    for future in as_completed(fs):
        print(future.result())

if DOWNLOAD_UVR5:
    run(["unzip", "-q", "-o", str(DL_DIR / "uvr5_weights.zip"),
         "-d", str(REPO_DIR / "tools/uvr5")])


# V5 files are distributed separately from pretrained_models.zip.
v5_dir = REPO_DIR / "GPT_SoVITS" / "pretrained_models" / "gsv-v5-pretrained"
v5_dir.mkdir(parents=True, exist_ok=True)

v5_files = {
    DL_DIR / "s2Gv5dev.pth": v5_dir / "s2Gv5dev.pth",
    DL_DIR / "v5-vocoder.pth": v5_dir / "vocoder.pth",
}

for source, destination in v5_files.items():
    if not source.exists():
        raise FileNotFoundError(f"Downloaded V5 asset missing: {source}")

    if not destination.exists() or destination.stat().st_size != source.stat().st_size:
        print(f"[install] {destination.name}", flush=True)
        shutil.copyfile(source, destination)


t = elapsed("parallel asset extraction + V5 install", t)


print("[check] Python syntax for WebUI/emotion pipeline", flush=True)

syntax_targets = [
    REPO_DIR / "webui.py",
    REPO_DIR / "tools/asr/config.py",
    REPO_DIR / "tools/asr/fasterwhisper_asr.py",
    REPO_DIR / "tools/asr/funasr_asr.py",
    REPO_DIR / "tools/emotion_training_pipeline.py",
    REPO_DIR / "tools/auto_emotion_sidecar.py",
    REPO_DIR / "tools/emotion_dataset_builder.py",
    REPO_DIR / "tools/run_emotion_v5_preprocess.py",
    REPO_DIR / "tools/train_emotion_controller.py",
    REPO_DIR / "GPT_SoVITS/inference_webui.py",
    REPO_DIR / "GPT_SoVITS/prepare_datasets/1-get-text.py",
    REPO_DIR / "GPT_SoVITS/prepare_datasets/3-get-semantic.py",
    REPO_DIR / "GPT_SoVITS/text/chinese.py",
    REPO_DIR / "GPT_SoVITS/text/chinese2.py",
    REPO_DIR / "GPT_SoVITS/text/korean.py",
]

run(
    [
        sys.executable,
        "-m",
        "py_compile",
        *[str(path) for path in syntax_targets],
    ],
    cwd=REPO_DIR,
)


# 6. Verify only in a NEW Python process.
verify_script = r"""
import importlib
import sys
import traceback

mods = [
    ("numpy", "NumPy"),
    ("faster_whisper", "faster-whisper"),
    ("funasr", "FunASR"),
    ("modelscope", "ModelScope"),
    ("onnxruntime", "ONNX Runtime"),
    ("tensorboard", "TensorBoard"),
    ("ctranslate2", "CTranslate2"),
    ("torchcodec", "TorchCodec"),
    ("gradio", "Gradio"),
    ("transformers", "Transformers"),
    ("pytorch_lightning", "PyTorch Lightning"),
    ("pyopenjtalk", "OpenJTalk"),
    ("huggingface_hub", "Hugging Face Hub"),
    ("hf_xet", "HF Xet"),
]
failed = []
for module_name, label in mods:
    try:
        mod = importlib.import_module(module_name)
        print(f"[OK] {label}: {getattr(mod, '__version__', 'installed')}")
    except Exception:
        failed.append((label, traceback.format_exc()))

if failed:
    for label, tb in failed:
        print(f"\n[{label}]\n{tb}", file=sys.stderr)
    raise SystemExit(1)
"""

verify = run([sys.executable, "-c", verify_script],
             cwd=REPO_DIR, check=False, capture=True, quiet=True)
print(verify.stdout)
if verify.returncode != 0:
    print(verify.stderr)
    if "numpy.dtype size changed" in verify.stderr:
        print("[repair] Reinstalling ABI-sensitive wheels in the fresh environment.")
        numpy_repair = (
            "numpy>=2.4,<3"
            if sys.version_info >= (3, 13)
            else "numpy==1.26.4"
        )
        run([
            UV, "pip", "install", "--system", "--reinstall",
            numpy_repair,
            "pandas>=2.2.2,<4",
            "scipy>=1.13,<2",
            "scikit-learn>=1.5,<2",
        ], env=uv_env)
        verify = run([sys.executable, "-c", verify_script],
                     cwd=REPO_DIR, check=False, capture=True, quiet=True)
        print(verify.stdout)
        if verify.returncode != 0:
            print(verify.stderr)
            raise RuntimeError("Fresh-process dependency verification still failed.")
    else:
        raise RuntimeError("Fresh-process dependency verification failed.")

needed = [
    REPO_DIR / "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    REPO_DIR / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth",
    REPO_DIR / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth",
]
missing = [str(p) for p in needed if not p.exists()]
if missing:
    raise FileNotFoundError("V5dev assets missing:\n" + "\n".join(missing))

for emotion_name, emotion_dir in EMOTION_MODEL_DIRS.items():
    emotion_weight = emotion_dir / "model.pt"
    if emotion_weight.is_file():
        print(
            f"[OK] emotion2vec+ {emotion_name}:",
            emotion_weight,
            f"({emotion_weight.stat().st_size / 1024**3:.2f} GiB)",
        )
    else:
        print(
            f"[warn] emotion2vec+ {emotion_name} is not prefetched; "
            "automatic labeling may use its hub fallback."
        )

smoke = r"""
import sys
from pathlib import Path
repo = Path("/content/GPT-SoVITS")
sys.path.insert(0, str(repo / "GPT_SoVITS"))
import numpy, torch, torchaudio
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from module.models import Generator
from module.models_v5 import V5_VERSIONS
from text.cleaner import clean_text
if sys.version_info >= (3, 14):
    from importlib.metadata import version
    from packaging.version import Version
    if Version(version("torch").split("+", 1)[0]) < Version("2.10"):
        raise RuntimeError("Python 3.14 requires PyTorch >=2.10")

phones, _, normalized = clean_text("안녕하세요", "ko", "v2")
if not phones:
    raise RuntimeError("Korean G2P smoke test returned no phones")

print("V5_SMOKE_OK")
print("python=", sys.version.split()[0])
print("numpy=", numpy.__version__)
print("torch=", torch.__version__)
print("korean_g2p=", len(phones), normalized)
print("versions=", sorted(V5_VERSIONS))
"""
run([sys.executable, "-c", smoke], cwd=REPO_DIR)

elapsed("fresh-process verification", t)
print(f"\n[setup] TOTAL: {time.perf_counter() - T0:.1f}s")
print("[setup] Full GPT-SoVITS install ready.")
