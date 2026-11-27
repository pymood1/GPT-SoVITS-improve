"""Paired, speaker-clustered exploratory analysis of completed S1/S2 screening.

Two seeds and multiple utterances per speaker are repeated measurements,
not independent speakers. No automatic deployment or significance claim.
Uses stdlib only; accepts the original aggregate JSON, no audio required.
"""
import argparse
import itertools
import json
import math
import random
import statistics as st
from pathlib import Path

BASE={"en":"ar","ko":"cfg13"}
METRICS={
    "cer":(-1,lambda r:r["char_errors"]/r["char_count"]),
    "speaker_cosine":(1,lambda r:r["speaker_cosine"]),
    "abs_log_duration":(-1,lambda r:abs(math.log(r["duration_ratio"]))),
    "periodicity":(1,lambda r:r["periodicity"]["median_periodicity"]),
    "active_hf":(-1,lambda r:r["noise"]["active_hf_ratio"]),
    "generation_seconds":(-1,lambda r:r.get("generation_seconds")),
}
def mean(v):return st.mean(v) if v else None
def usable(v):return v is not None and math.isfinite(v)
def interval(values,seed=20261127,n=10000):
    if len(values)<2:return None
    rng=random.Random(seed)
    boots=sorted(st.mean(rng.choices(values,k=len(values))) for _ in range(n))
    return [boots[int(.025*n)],boots[min(n-1,int(.975*n))]]
def exact_sign_flip(values):
    if len(values)>16 or len(values)<2:return None
    actual=abs(st.mean(values))
    return sum(abs(st.mean(s*v for s,v in zip(signs,values)))>=actual-1e-12
               for signs in itertools.product((-1,1),repeat=len(values)))/(2**len(values))

def analyze(report):
    if report["status"]!="completed":raise ValueError("Incomplete run")
    by={};seeds=set()
    for rep in report["replications"]:
        seed=rep["seed"]
        if seed in seeds:raise ValueError("Duplicate seed")
        seeds.add(seed)
        for row in rep["rows"]:
            lang=row["language"];arm=row["arm"]
            key=(seed,row["reference_index"],row["target_index"])
            bucket=by.setdefault(lang,{}).setdefault(arm,{})
            if key in bucket:raise ValueError("Duplicate paired row")
            bucket[key]=row
    if sum(len(a) for b in by.values() for a in b.values())!=report["actual_cases"]:
        raise ValueError("Case count mismatch")
    out={}
    for lang,arms in by.items():
        base=arms[BASE[lang]];keys=sorted(base)
        speakers=sorted({k[1] for k in keys});result={}
        for arm,rows in arms.items():
            if set(rows)!=set(keys):raise ValueError("Unmatched cases: "+arm)
            rr={}
            for name,(direction,get) in METRICS.items():
                paired=[]
                for k in keys:
                    x,y=get(rows[k]),get(base[k])
                    if usable(x) and usable(y):paired.append((k,x,y,direction*(x-y)))
                clusters=[mean([d for k,x,y,d in paired if k[1]==sp]) for sp in speakers]
                clusters=[x for x in clusters if x is not None]
                sd={str(seed):mean([d for k,x,y,d in paired if k[0]==seed]) for seed in sorted(seeds)}
                rr[name]=dict(arm_mean=mean([x for k,x,y,d in paired]),baseline_mean=mean([y for k,x,y,d in paired]),
                    improvement=mean(clusters),speaker_deltas=clusters,seed_deltas=sd,
                    speakers_improved=sum(x>1e-12 for x in clusters),speakers_regressed=sum(x< -1e-12 for x in clusters),
                    n_speakers=len(clusters),n_pairs=len(paired),exploratory_bootstrap95=interval(clusters),
                    unadjusted_sign_flip_p=exact_sign_flip(clusters))
            rr["micro_cer"]=sum(x["char_errors"] for x in rows.values())/sum(x["char_count"] for x in rows.values())
            rr["short"]=sum(x["duration_ratio"]<.5 for x in rows.values())
            rr["long"]=sum(x["duration_ratio"]>2 for x in rows.values())
            rr["caps"]=sum(bool(x.get("cap_hit",False)) for x in rows.values())
            result[arm]=rr
        # Screen, not select: CER and identity only; other metrics are trade-offs.
        def dominates(a,b):
            return (a["micro_cer"]<=b["micro_cer"] and a["speaker_cosine"]["arm_mean"]>=b["speaker_cosine"]["arm_mean"]
                    and (a["micro_cer"]<b["micro_cer"] or a["speaker_cosine"]["arm_mean"]>b["speaker_cosine"]["arm_mean"]))
        frontier=[a for a in result if not any(dominates(b,result[a]) for k,b in result.items() if k!=a)]
        out[lang]=dict(baseline=BASE[lang],speaker_count=len(speakers),seed_count=len(seeds),
            arms=result,cer_identity_pareto=frontier)
    return dict(code_sha=report.get("code_sha"),actual_cases=report["actual_cases"],languages=out,
        caveat="Exploratory paired speaker-level comparisons, small clusters, reused seeds/utterances, unadjusted multiple comparisons. HF/periodicity are not proof of perceived noise. No deployment decision.")

def selftest():
    rows=[]
    for speaker in (1,2):
        for arm,errors in (("ar",4),("candidate",2)):
            rows.append(dict(language="en",arm=arm,reference_index=speaker,target_index=1,char_errors=errors,char_count=10,
                speaker_cosine=.8,duration_ratio=1.,periodicity={"median_periodicity":.9},noise={"active_hf_ratio":.01}))
    report=dict(status="completed",actual_cases=4,replications=[dict(seed=1,rows=rows)])
    out=analyze(report)["languages"]["en"]
    assert out["speaker_count"]==2 and out["cer_identity_pareto"]==["candidate"]
    c=out["arms"]["candidate"]["cer"]
    assert abs(c["improvement"]-.2)<1e-12 and c["speakers_improved"]==2
    assert c["unadjusted_sign_flip_p"]==.5
    assert out["arms"]["candidate"]["generation_seconds"]["improvement"] is None
    print("Paired speaker grouping, direction, missing metrics, sign-flip and Pareto tests passed")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true");p.add_argument("--source",type=Path);p.add_argument("--output",type=Path);a=p.parse_args()
    if a.selftest:selftest()
    else:
        s=json.dumps(analyze(json.loads(a.source.read_text())),indent=2,allow_nan=False)
        if a.output:a.output.write_text(s+"\n")
        else:print(s)
