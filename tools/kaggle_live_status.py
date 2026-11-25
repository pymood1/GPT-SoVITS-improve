"""Read private Kaggle logs in Actions; publish only fixed-schema progress to PR.

Never print raw CLI output, transcripts, exceptions, or credentials.
The GitHub token stays in Actions and is stripped from Kaggle subprocess env.
"""
import datetime
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LABELS = frozenset(("base", "A_lr100", "B_lr030", "s1_only", "s2_only", "combined",
    "rank16", "rank32", "mix025", "mix050", "mix075", "mix125", "cfg100", "cfg160",
    "soft000", "soft005", "soft010", "soft025", "soft050", "soft100", "soft000b",
    "mix065", "mix085", "gen_base", "gen_hard", "gen_soft010", "gen_hardb",
    "gen_soft010b", "gen_soft025", "A_cfm_off", "B_cfm_on", "A_wd001", "B_wd1",
    "S1_v_o", "S1_qkv_o", "S2_rank16", "S2_rank32"))
PHASES = frozenset(("bootstrap", "preprocess", "training", "model_load", "synthesis", "asr",
                    "speaker", "finished", "failed"))
STATES = frozenset(("queued", "running", "complete", "completed", "error", "failed",
                    "cancel_requested", "cancel_acknowledged", "cancelled", "canceled"))
TERMINAL = STATES - {"queued", "running", "cancel_requested"}
# Some CLI versions prefix lines with a relative timestamp or ISO timestamp.
PREFIX = r"^(?:(?:\d+(?:\.\d+)?s?|\d{4}-\d\d-\d\d[T ][0-9:.+-]+Z?)\s+(?:\|\s*)?)?"
EVENT = re.compile(PREFIX + r"\[gsv-progress\] (gpu[01]|parent) ([a-z_]+) ([A-Za-z0-9_]+) (\d{1,6})/(\d{1,6})$")
LEGACY = re.compile(PREFIX + r"\[quality\] ([A-Za-z0-9_]+): completed (\d{1,6}) samples$")


def parse_progress(raw):
    workers, completed = {}, {}
    for line in raw.splitlines():
        m = EVENT.fullmatch(line.strip())
        if m:
            worker, phase, label, done, total = m.groups()
            done, total = int(done), int(total)
            if phase in PHASES and (label == "all" or label in LABELS) and 0 <= done <= total <= 100000:
                workers[worker] = {"phase": phase, "label": label, "done": done, "total": total}
            continue
        m = LEGACY.fullmatch(line.strip())
        if m and m[1] in LABELS and int(m[2]) <= 100000:
            completed[m[1]] = int(m[2])
    return {"workers": workers, "completed_synthesis": completed}


def cli(args, timeout=35):
    env = {k: v for k, v in os.environ.items() if k not in ("GH_TOKEN", "GITHUB_TOKEN")}
    try:
        p = subprocess.run(["kaggle", "kernels", *args], capture_output=True,
                           text=True, errors="replace", timeout=timeout, env=env)
        return (p.returncode == 0, p.stdout if p.returncode == 0 else "")
    except (OSError, subprocess.TimeoutExpired):
        return False, ""


def kernel_status(raw):
    m = re.search(r'KernelWorkerStatus[.]([A-Z_]+)', raw)
    state = m[1].lower() if m else "unknown"
    return state if state in STATES else "unknown"


def github(method, path, data=None):
    token = os.environ["GH_TOKEN"]
    req = urllib.request.Request("https://api.github.com" + path,
        data=None if data is None else json.dumps(data).encode(),
        method=method, headers={"Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json", "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.load(response)


def publish(kernel, state, snapshot, state_file):
    repo = os.environ["GITHUB_REPOSITORY"]
    pr = os.environ["RESEARCH_PR"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or not pr.isdecimal():
        raise ValueError("Invalid PR target")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    body = "<!-- gsv-live-status " + kernel + " -->\n" + (
        "### Kaggle research progress\n\n"
        + "Updated UTC: " + now + "\n\n"
        + "Kernel: " + kernel + "\n\n"
        + "Status: " + state + "\n\n"
        + "Sanitized progress (no private logs):\n\n"
        + "\x60\x60\x60json\n" + json.dumps(snapshot, indent=2) + "\n\x60\x60\x60")
    base = "/repos/" + repo + "/issues"
    if state_file.is_file():
        cid = int(state_file.read_text())
        github("PATCH", base + "/comments/" + str(cid), {"body": body})
    else:
        result = github("POST", base + "/" + pr + "/comments", {"body": body})
        state_file.write_text(str(int(result["id"])))


def main():
    kernel = sys.argv[1]
    if not re.fullmatch(r"pymood1/gpt-sovits-research-[0-9]+", kernel):
        raise ValueError("Invalid research kernel")
    state_file = Path(sys.argv[2])
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    deadline = time.monotonic() + max(0, min(duration, 14400))
    last_progress = None
    last_change = time.monotonic()
    while True:
        ok, status = cli(["status", kernel])
        state = kernel_status(status) if ok else "unknown"
        ok, raw = cli(["logs", kernel])
        parsed = parse_progress(raw) if ok else None
        if parsed is not None and parsed != last_progress:
            last_change = time.monotonic()
            last_progress = parsed
        snapshot = dict(last_progress or {"workers": {}, "completed_synthesis": {}})
        snapshot["log_read"] = "ok" if ok else "unavailable"
        snapshot["seconds_since_progress_change"] = int(time.monotonic() - last_change)
        print("SAFE_PROGRESS", state, json.dumps(snapshot), flush=True)
        try:
            publish(kernel, state, snapshot, state_file)
        except Exception:
            print("SAFE_PROGRESS comment unavailable", flush=True)
        if state in TERMINAL or time.monotonic() >= deadline:
            return 0
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
