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
WORKER = re.compile(r"(ddp|cache[01]|eval[01]|pseudo[01]|worker[01]|asr-fp32|speaker|(?:en|ko)-(?:asr|speaker)|parent)")
EVENTS = {"pseudo_pair","asr_progress","cache_trial","layer_epoch","layer_result","train_step","worker_progress","phase"}
ARMS = {"teacher","round12","round14","student","prior_student","frozen","native","text_pad","eos_mask","combined","combined_weight4","eos2","eos4","eos8","eos4_ctc","eos4_ctc_strong","real","oracle","base","continued","contrastive","gain","late_cfg","gain_late","zero","bert","bert_reverse","native09","full09","full13","late09","late13","nearest","word","pooled","pad_native","pad_none","pad300","pad_trim","ar","nar_one","nar_iter","nar_oracle_len","nar_repair","noise0875","noise065","noise100","noise110","ar_soft","ar_band","ar_exact","ar_greedy","ar_lowtemp","ar_norep","pre_one","pre_iter","pre_local","pre_repair","cfg10","cfg13","cfg16","cfg19","sway05","sway10","euler64","mid16","gain13","trim13"}
METRICS = {"ce","teacher_ce","ce_ratio","kl","top1_agreement","target_accuracy","teacher_accuracy","tokens"}
COUNTS = {"epoch","tokens","step","index","accepted_count","gate_streak","attempts","completed","total","reference_index","text_index","elapsed_seconds"}
CACHE = {"full_seconds","cached_seconds","max_abs","mean_abs","parity","top1_matches","cache_bytes","sampled_equal","sampled_full_seconds","sampled_cached_seconds","sampled_full_tokens","sampled_cached_tokens"}

def scalar(v):
    return type(v) in (int,float,bool) and (type(v) is not float or math.isfinite(v)) and abs(v)<=1e12

def numbers(row, keys):
    return {k:v for k,v in row.items() if k in keys and scalar(v)} if isinstance(row,dict) else {}

def safe_latest(row):
    if not isinstance(row,dict):
        return None
    out=numbers(row,COUNTS|{"accepted","scale"})
    for key,allowed in (("event",EVENTS),("arm",ARMS),("status",{"ok","failed","training","synthesized","test","validation","ko_guard","running","completed"}),("split",{"train","validation","test","ko_guard"}),("language",{"en","ko","all_ko"}),("stage",{"setup","corpus","model_load","training","synthesis","asr","speaker","finished","failed"}),("backend",{"ct2_cuda_float32","torch_cuda_float32"})):
        if isinstance(row.get(key),str) and row[key] in allowed:
            out[key]=row[key]
    if "validation" in row:
        out["validation"]=numbers(row["validation"],METRICS)
    if isinstance(row.get("models"),dict):
        out["models"]={k:numbers(v,CACHE) for k,v in row["models"].items() if k in ARMS and isinstance(v,dict)}
    return out or None

# Accept only structural log prefixes and allowlisted JSON fields. No raw text is published.
LINE = re.compile(r"^\s*(?:(?:\d+(?:\.\d+)?s?|\d{4}-\d\d-\d\d[T ][0-9:.+-]+Z?)\s*(?:\|\s*)?)?(?:\d+\s*\|\s*)?\[scale\]\s+(\{.*\})\s*$")
def log_lines(raw):
    # SDK versions may return an array/object of structured log messages.
    def visit(value,depth=0):
        if depth>6:return
        if isinstance(value,str):
            yield from value.splitlines()
        elif isinstance(value,list):
            for item in value:yield from visit(item,depth+1)
        elif isinstance(value,dict):
            for k in ("logs","log","messages","message","text","data","entries","content"):
                if k in value:yield from visit(value[k],depth+1)
    try:value=json.loads(raw)
    except (ValueError,RecursionError):value=None
    if isinstance(value,(list,dict,str)):
        yield from visit(value)
    else:
        for line in raw.splitlines():
            try:obj=json.loads(line)
            except (ValueError,RecursionError):obj=None
            if isinstance(obj,(dict,list,str)):
                yield from visit(obj)
            else:yield line

def parse(raw):
    workers={}
    for line in log_lines(raw):
        m=LINE.fullmatch(line)
        if not m:continue
        try:row=json.loads(m[1])
        except ValueError:continue
        if not isinstance(row,dict):continue
        for key,v in row.items():
            if not WORKER.fullmatch(key) or not isinstance(v,dict):continue
            code=v.get("exit")
            workers[key]={"exit":code if type(code) is int and -255<=code<=255 else None,
                          "latest":safe_latest(v.get("latest"))}
    return workers


# Backup numeric-only terminal metrics, independent of output file listing.
RESULT_LINE = re.compile(r"^\s*(?:(?:\d+(?:\.\d+)?s?|\d{4}-\d\d-\d\d[T ][0-9:.+-]+Z?)\s*(?:\|\s*)?)?(?:\d+\s*\|\s*)?\[s1-result\]\s+(\{.*\})\s*$")
SUMMARY_FIELDS = {"cases","cer","speaker_cosine","median_duration_ratio","median_abs_log_duration_error","short","long","cap_hits","median_periodicity","mean_active_hf"}
def safe_summary(row):
    if not isinstance(row,dict):return None
    sha=row.get("code_sha");lang=row.get("language");arm=row.get("arm");seed=row.get("seed")
    if not isinstance(sha,str) or not re.fullmatch(r"[0-9a-f]{40}",sha):return None
    if lang not in ("en","ko") or not isinstance(arm,str) or arm not in ARMS or type(seed) is not int or not 0<=seed<=15:return None
    values=numbers(row.get("metrics"),SUMMARY_FIELDS)
    if type(values.get("cases")) is not int or not 0<values["cases"]<=100000:return None
    if not {"cer","speaker_cosine"}<=values.keys():return None
    return dict(code_sha=sha,language=lang,seed=seed,arm=arm,metrics=values)

def parse_results(raw):
    rows={}
    for line in log_lines(raw):
        m=RESULT_LINE.fullmatch(line)
        if not m:continue
        try: row=safe_summary(json.loads(m[1]))
        except (ValueError,TypeError):continue
        if row:
            key=(row["code_sha"],row["language"],row["seed"],row["arm"])
            rows[key]=row
    return [rows[k] for k in sorted(rows)][:120]

def emit_summary(report):
    if report.get("status")!="completed":return
    for key,metrics in report.get("waveform_summary",{}).items():
        m=re.fullmatch(r"(en|ko)-seed(\d+)-([a-z0-9_]+)",key)
        if not m:continue
        row=safe_summary(dict(code_sha=report.get("code_sha"),language=m[1],seed=int(m[2]),arm=m[3],metrics=metrics))
        if row:print("[s1-result]",json.dumps(row,allow_nan=False,separators=(",",":")),flush=True)

def snapshot(kernel):
    ok,raw=cli(["status",kernel])
    state=kernel_status(raw) if ok else "unknown"
    ok,raw=cli(["logs",kernel])
    data={"log_read":"ok" if ok else "unavailable","workers":parse(raw) if ok else {}}
    if ok and state in TERMINAL:
        results=parse_results(raw)
        if results:data["aggregate_backup"]=results
    return state,data

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
    structured=json.dumps({"logs":[{"message":'2026-11-27T09:23:00Z | 12 | [scale] {"en-asr":{"exit":null,"latest":{"event":"asr_progress","completed":7,"total":48,"private":"secret"}}}'}]})
    assert parse(structured)=={"en-asr":{"exit":None,"latest":{"event":"asr_progress","completed":7,"total":48}}}
    assert parse('[scale] {"eval0":{"exit":null,"latest":{"arm":"contrastive","status":"training","reference_index":20,"text_index":160,"language":"en","loss":1.2}}}')["eval0"]["latest"]=={"arm":"contrastive","status":"training","reference_index":20,"text_index":160,"language":"en"}
    assert parse('untrusted prefix [scale] {"ko-asr":{"exit":0}}')=={}
    row={"code_sha":"a"*40,"language":"en","seed":0,"arm":"ar","metrics":{"cases":8,"cer":.2,"speaker_cosine":.8,"secret":"never"},"private":"never"}
    saved=parse_results("123.4s | 1 | [s1-result] "+json.dumps(row))
    assert len(saved)==1 and "private" not in saved[0] and "secret" not in saved[0]["metrics"]
    assert parse_results("untrusted [s1-result] "+json.dumps(row))==[]
    print("S1 safe observer and numeric backup tests passed")

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
