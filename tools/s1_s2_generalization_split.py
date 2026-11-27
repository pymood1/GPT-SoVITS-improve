"""Speaker-disjoint partition and quality-vs-diversity experiment design.
Input: JSON list of metadata rows, no waveform or model access.
Each row: speaker_id, text, sha256, seconds, quality_score (higher better).
Quality score must come from independent validated data audit, not test-model
performance. No training or inference here.
"""
import argparse,hashlib,json,math,unicodedata
from collections import defaultdict
from pathlib import Path

def normalize(s):
    return "".join(c for c in unicodedata.normalize("NFKC",s).casefold() if c.isalnum())

def rank(seed,key):
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()

def partition(rows,seed=20261127):
    speakers=sorted({str(r["speaker_id"]) for r in rows},key=lambda s:rank(seed,s))
    if len(speakers)<10:raise ValueError("At least ten speakers required")
    n=max(2,len(speakers)//5)
    speaker_sets={"test":set(speakers[:n]),"validation":set(speakers[n:2*n]),"train":set(speakers[2*n:])}
    out={k:[] for k in speaker_sets};seen_text=set();seen_audio=set()
    # Holdout has priority; repeated sentences/audio removed across partitions.
    for split in ("test","validation","train"):
        for r in sorted(rows,key=lambda r:rank(seed,r["sha256"])):
            if str(r["speaker_id"]) not in speaker_sets[split]:continue
            text=normalize(r["text"]);digest=r["sha256"]
            if not text or not digest or text in seen_text or digest in seen_audio:continue
            if not math.isfinite(r["seconds"]) or not 3<=r["seconds"]<=10:continue
            if not math.isfinite(r["quality_score"]):continue
            out[split].append(r);seen_text.add(text);seen_audio.add(digest)
    if any(len({str(r["speaker_id"]) for r in out[k]})<2 for k in out):
        raise ValueError("Too few usable speakers after deduplication")
    return out

def arms(train,budget_seconds,quality_floor):
    # Same quality-qualified pool; different selection order.
    pool=[r for r in train if r["quality_score"]>=quality_floor]
    groups=defaultdict(list)
    for r in pool:groups[str(r["speaker_id"])].append(r)
    if len(groups)<2:raise ValueError("Insufficient qualified speakers")
    quality=sorted(pool,key=lambda r:(-r["quality_score"],r["sha256"]))
    for s in groups:groups[s].sort(key=lambda r:(-r["quality_score"],r["sha256"]))
    diverse=[]
    order=sorted(groups)
    for i in range(max(map(len,groups.values()))):
        diverse.extend(groups[s][i] for s in order if i<len(groups[s]))
    def fit(order):
        rows=[];seconds=0.
        for r in order:
            if seconds+r["seconds"]<=budget_seconds:
                rows.append(r);seconds+=r["seconds"]
        if seconds<budget_seconds-10:raise ValueError("Budget cannot be filled")
        return {"rows":rows,"seconds":seconds,"speakers":len({str(r["speaker_id"]) for r in rows}),
                "mean_quality":sum(r["quality_score"] for r in rows)/len(rows)}
    a,b=fit(quality),fit(diverse)
    if abs(a["seconds"]-b["seconds"])>10:raise ValueError("Unmatched duration")
    return {"quality_ranked":a,"speaker_balanced":b,
            "caveat":"Quality may still differ; report mean score and stratify before causal attribution. Balanced arm is not guaranteed to improve."}

def selftest():
    rows=[dict(speaker_id=str(s),text=f"unique {s} {i}",sha256=f"{s}-{i}",seconds=5.,quality_score=1-s*.01)
          for s in range(12) for i in range(20)]
    split=partition(rows)
    sets=[{r["speaker_id"] for r in v} for v in split.values()]
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    x=arms(split["train"],60.,.8)
    assert x["quality_ranked"]["seconds"]==x["speaker_balanced"]["seconds"]==60
    assert x["speaker_balanced"]["speakers"]>x["quality_ranked"]["speakers"]
    print("Generalization split selftest passed")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("metadata",nargs="?",type=Path);p.add_argument("--output",type=Path)
    p.add_argument("--budget-seconds",type=float);p.add_argument("--quality-floor",type=float)
    p.add_argument("--selftest",action="store_true")
    a=p.parse_args()
    if a.selftest:selftest()
    else:
        if a.metadata is None or a.output is None or a.budget_seconds is None or a.quality_floor is None:p.error("metadata, output, budget and quality floor required")
        splits=partition(json.loads(a.metadata.read_text()))
        result={"splits":splits,"arms":arms(splits["train"],a.budget_seconds,a.quality_floor)}
        a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
