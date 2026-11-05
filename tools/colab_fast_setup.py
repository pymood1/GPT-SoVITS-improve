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
HF_BASE = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"
ASSETS = {
    "pretrained_models.zip": f"{HF_BASE}/pretrained_models.zip",
    "G2PWModel.zip": f"{HF_BASE}/G2PWModel.zip",
    "nltk_data.zip": f"{HF_BASE}/nltk_data.zip",
    "open_jtalk_dic_utf_8-1.11.tar.gz": f"{HF_BASE}/open_jtalk_dic_utf_8-1.11.tar.gz",
}
if DOWNLOAD_UVR5:
    ASSETS["uvr5_weights.zip"] = f"{HF_BASE}/uvr5_weights.zip"


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
if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
    raise RuntimeError("Use Colab Runtime Version 2026.07 (Python 3.12).")

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
})
run([UV, "--version"], env=uv_env)
t = elapsed("uv bootstrap", t)


# 3. Keep the COMPLETE upstream requirements.
#    OpenCC remains installed; only force-source compilation is relaxed.
#    GPT-SoVITS requires NumPy <2, so use the latest 1.x ABI explicitly.
repo_req = REPO_DIR / "requirements.txt"
repo_extra = REPO_DIR / "extra-req.txt"
lines = repo_req.read_text(encoding="utf-8").splitlines()
lines = [line for line in lines if line.strip() != "--no-binary=opencc"]
lines = ["numpy==1.26.4" if line.strip() == "numpy<2.0" else line for line in lines]

full_req = Path("/content/gpt_sovits_full_requirements.txt")
full_req.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("[setup] preparing full requirements...", flush=True)
print("Full requirements retained:", sum(
    1 for x in lines if x.strip() and not x.lstrip().startswith("#")
))
print("extra-req.txt retained:", repo_extra.read_text(encoding="utf-8").strip())
print("NumPy ABI target: 1.26.4")
print("No GPT-SoVITS dependency is intentionally excluded.")


# 4. Download all original installer assets while uv installs all Python deps.
def download_one(name, url):
    dest = DL_DIR / name
    part = DL_DIR / f"{name}.part"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[download] reuse {name} ({dest.stat().st_size / 1024**2:.1f} MiB)")
        return dest

    print(f"[download] start {name}")
    cmd = [
        "curl", "-L", "--fail",
        "--retry", "8", "--retry-delay", "2",
        "--connect-timeout", "20",
        "--speed-time", "30", "--speed-limit", "1024",
        "-C", "-", "-o", str(part), url,
    ]
    subprocess.run(cmd, check=True)
    part.replace(dest)
    print(f"[download] done  {name} ({dest.stat().st_size / 1024**2:.1f} MiB)")
    return dest


pip_cmd = [
    UV, "pip", "install", "--system",
    "-r", str(full_req),
    "torchcodec",
]
print("[setup] installing full Python dependencies while downloading assets...", flush=True)
print("+", " ".join(pip_cmd), flush=True)
pip_proc = subprocess.Popen(pip_cmd, cwd=str(REPO_DIR), env=uv_env, text=True)

errors = []
with ThreadPoolExecutor(max_workers=min(4, len(ASSETS))) as pool:
    futures = {pool.submit(download_one, n, u): n for n, u in ASSETS.items()}
    for future in as_completed(futures):
        name = futures[future]
        try:
            future.result()
        except Exception as exc:
            errors.append((name, repr(exc)))

rc = pip_proc.wait()
if rc:
    raise subprocess.CalledProcessError(rc, pip_cmd)

# Same behavior as upstream install.sh for faster-whisper.
run([
    UV, "pip", "install", "--system",
    "-r", str(repo_extra), "--no-deps",
], cwd=REPO_DIR, env=uv_env)

if errors:
    raise RuntimeError("Asset download failed:\n" + "\n".join(f"{n}: {e}" for n, e in errors))
t = elapsed("full dependencies + parallel downloads", t)


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
    marker = Path(sys.prefix) / "nltk_data"
    if marker.exists():
        return "[extract] NLTK already present"
    run(["unzip", "-q", "-o", str(DL_DIR / "nltk_data.zip"),
         "-d", sys.prefix], quiet=True)
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
t = elapsed("parallel asset extraction", t)


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
        run([
            UV, "pip", "install", "--system", "--reinstall",
            "numpy==1.26.4",
            "pandas>=2.2.2,<3",
            "scipy>=1.12,<2",
            "scikit-learn>=1.4,<2",
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
print("V5_SMOKE_OK")
print("numpy=", numpy.__version__)
print("torch=", torch.__version__)
print("versions=", sorted(V5_VERSIONS))
"""
run([sys.executable, "-c", smoke], cwd=REPO_DIR)

elapsed("fresh-process verification", t)
print(f"\n[setup] TOTAL: {time.perf_counter() - T0:.1f}s")
print("[setup] Full GPT-SoVITS install ready.")
