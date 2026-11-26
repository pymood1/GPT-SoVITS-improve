"""Read-only S1 Kaggle observer: allowlisted progress, never raw logs."""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

from tools.kaggle_live_status import cli, kernel_status, publish, TERMINAL

KERNEL = re.compile(r"pymood1/gpt-sovits-s1-next-[0-9]{1,18}")
WORKER = re.compile(r"(ddp|cache[01]|eval[01]|pseudo[01]|worker[01]|asr-fp32|speaker)")
EVENTS = {"pseudo_pair","asr_progress","cache_trial","layer_epoch","layer_result","train_step"}
ARMS = {"teacher","round12","round14","student","prior_student"}
METRICS = {"ce","teacher_ce","ce_ratio","kl","top1_agreement","target_accuracy","teacher_accuracy","tokens"}
COUNTS = {"epoch","tokens","step","index","accepted_count","gate_streak","attempts","completed","total","reference_index","text_index"}
CACHE = {"full_seconds","cached_seconds","max_abs","mean_abs","parity","top1_matches","cache_bytes","sampled_equal","sampled_full_seconds","sampled_cached_seconds","sampled_full_tokens","sampled_cached_tokens"}

def scalar(v):
    return type(v) in (int,float,bool) and (type(v) is not float or math.isfinite(v)) and abs(v)<=1e12

def numbers(row, keys):
    return {k:v for k,v in row.items() if k in keys and scalar(v)} if isinstance(row,dict) else {}

def safe_latest(row):
    if not isinstance(row,dict):
        return None
    out=numbers(row,COUNTS|{"accepted","scale"})
    for key,allowed in (("event",EVENTS),("arm",ARMS),("status",{"ok","failed"}),("split",{"train","validation"}),("backend",{"ct2_cuda_float32","torch_cuda_float32"})):
        if isinstance(row.get(key),str) and row[key] in allowed:
            out[key]=row[key]
    if "validation" in row:
        out["validation"]=numbers(row["validation"],METRICS)
    if isinstance(row.get("models"),dict):
        out["models"]={k:numbers(v,CACHE) for k,v in row["models"].items() if k in ARMS and isinstance(v,dict)}
    return out or None

def parse(raw):
    workers={}
    for line in raw.splitlines():
        # Kaggle may prefix each line with timestamps and a line number.
        m=re.match(r"^\s*(?:(?:\d+(?:\.\d+)?s?\s*(?:\|\s*)?)(?:\d+\s*\|\s*)?)?\[scale\]\s+(\{.*\})\s*$",line)
        if not m:
            continue
        try:
            row=json.loads(m[1])
        except ValueError:
            continue
        if not isinstance(row,dict):
            continue
        for key,v in row.items():
            if not WORKER.fullmatch(key) or not isinstance(v,dict):
                continue
            code=v.get("exit")
            workers[key]={"exit":code if type(code) is int and -255<=code<=255 else None,
                          "latest":safe_latest(v.get("latest"))}
    return workers

def snapshot(kernel):
    ok,raw=cli(["status",kernel])
    state=kernel_status(raw) if ok else "unknown"
    ok,raw=cli(["logs",kernel])
    return state,{"log_read":"ok" if ok else "unavailable","workers":parse(raw) if ok else {}}

def selftest():
    raw='387.8s | 161 | [scale] {"ddp":{"exit":null,"latest":{"event":"layer_epoch","epoch":1,"validation":{"ce":2.84,"private":"secret"}}},"private":{"latest":{"epoch":99}}}'
    assert parse(raw)=={"ddp":{"exit":None,"latest":{"epoch":1,"event":"layer_epoch","validation":{"ce":2.84}}}}
    assert parse("private [scale] "+raw)=={}
    assert parse('[scale] {"ddp":{"exit":null,"latest":{"validation":{"ce":NaN}}}}')=={"ddp":{"exit":None,"latest":{"validation":{}}}}
    assert parse("[scale] []\n[scale] 0")=={}
    assert parse('[scale] {"ddp":null,"other":{"exit":0}}')=={}
    assert safe_latest({"arm":"secret","status":"secret","tokens":"secret"}) is None
    assert safe_latest({"models":{"student":{"parity":True,"private":"secret"}}})=={"models":{"student":{"parity":True}}}
    assert parse('[scale] {"eval0":{"exit":1,"latest":{"arm":"teacher","status":"failed"}}}')["eval0"]["exit"]==1
    print("S1 safe observer tests passed")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("kernel",nargs="?")
    p.add_argument("--publish",type=Path)
    p.add_argument("--duration",type=int,default=0)
    p.add_argument("--selftest",action="store_true")
    a=p.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.kernel or not KERNEL.fullmatch(a.kernel):
        p.error("Invalid S1 research kernel")
    end=time.monotonic()+max(0,min(a.duration,14400))
    while True:
        state,data=snapshot(a.kernel)
        print("S1_SAFE_PROGRESS",state,json.dumps(data,allow_nan=False),flush=True)
        if a.publish:
            try:
                publish(a.kernel,state,data,a.publish)
            except Exception:
                print("S1_SAFE_PROGRESS comment unavailable",flush=True)
        if state in TERMINAL or time.monotonic()>=end:
            return
        time.sleep(60)

if __name__=="__main__":
    main()
