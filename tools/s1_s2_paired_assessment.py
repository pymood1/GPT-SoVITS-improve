"""Predeclared paired factorial assessment; no automatic deployment.
Uses report from paired-s2-v6. No inference, no audio, no model changes.
"""
import argparse,json,math
from pathlib import Path
from statistics import mean
ARMS=("raw","raw-hard","gain","gain-hard","real","true-mel-vocoder","oracle-s2","v2ProPlus")
CANDIDATES=ARMS[1:4]

def assess(report):
    if report["status"]!="completed" or not report["same_speaker_real_target_verified"]:
        raise ValueError("Completed verified paired report required")
    if report["score_backends"]!={"asr":"ct2_cuda_float32","speaker":"cuda_fp32"}:
        raise ValueError("Strict FP32 scores required")
    rows=report["rows"]
    groups={}
    for row in rows:
        group=groups.setdefault((row["reference_index"],row["text_index"]),{})
        if row["arm"] in group:raise ValueError("Duplicate arm")
        group[row["arm"]]=row
    if len(groups)!=4 or len({k[0] for k in groups})!=2 or len(rows)!=32:
        raise ValueError("Unexpected paired pilot size")
    def metrics(r):
        d=r["spectral_to_real"]
        return dict(cer=r["char_errors"]/r["char_count"],sv=r["speaker_cosine"],
            mel=d["dtw_logmel_l1_db"],duration_abs=abs(math.log(d["duration_ratio"])),
            stretch=d["path_stretch"],periodicity=r["periodicity"]["median_periodicity"],
            periodic_fraction=r["periodicity"]["periodic_fraction"])
    per=[]
    for key,g in sorted(groups.items()):
        if set(g)!=set(ARMS) or len({g[a]["token_sha256"] for a in ARMS[:4]})!=1:
            raise ValueError("Missing arm or token mismatch")
        if len({r["char_count"] for r in g.values()})!=1:raise ValueError("Mismatched target")
        vals={a:metrics(r) for a,r in g.items()}
        per.append(dict(reference_index=key[0],text_index=key[1],values=vals,
                        cap_hit=any(g[a]["cap_hit"] for a in ARMS[:4])))
    summary={}
    for arm in ARMS:
        r=[g[arm] for g in groups.values()]
        v=[p["values"][arm] for p in per]
        summary[arm]={"cer":sum(x["char_errors"] for x in r)/sum(x["char_count"] for x in r),
                      "sv":mean(x["sv"] for x in v),"mel":mean(x["mel"] for x in v)}
    base=summary["raw"]
    candidates={}
    for arm in CANDIDATES:
        s=summary[arm]
        deltas={k:s[k]-base[k] for k in base}
        paired=[p["values"][arm]["mel"]-p["values"]["raw"]["mel"] for p in per]
        by_speaker={str(i):mean(p["values"][arm]["mel"]-p["values"]["raw"]["mel"] for p in per if p["reference_index"]==i) for i in sorted({p["reference_index"] for p in per})}
        # Conservative engineering screen, not statistical noninferiority.
        checks={"cer_no_worse":deltas["cer"]<=1e-12,"sv_no_worse":deltas["sv"]>=-1e-12,
                "mel_better":deltas["mel"]<-1e-12,"mel_better_both_speakers":all(v<0 for v in by_speaker.values()),
                "no_token_cap":not any(p["cap_hit"] for p in per)}
        candidates[arm]={"delta":deltas,"mel_pair_deltas":paired,"mel_by_speaker":by_speaker,"checks":checks,
                         "eligible_for_expansion":all(checks.values())}
    return {"pairs":per,"summary":summary,"candidates":candidates,
            "decision":"Expansion screen only; no significance or deployment claim. Blinded listening required.",
            "periodicity_interpretation":"Compare coverage and proximity to real target, not maximize periodicity; voiced selection can differ."}

def selftest():
    rows=[]
    for i in (1,2):
        for j in (1,2):
            for a in ARMS:
                rows.append(dict(reference_index=i,text_index=j,arm=a,token_sha256="fixed",cap_hit=False,
                    char_count=20,char_errors=2 if a=="gain" else 1,speaker_cosine=.8,
                    spectral_to_real=dict(dtw_logmel_l1_db=4 if a in ("gain","raw-hard") else 5,duration_ratio=1,path_stretch=1),
                    periodicity=dict(median_periodicity=.8,periodic_fraction=.5)))
    r=assess(dict(status="completed",same_speaker_real_target_verified=True,score_backends={"asr":"ct2_cuda_float32","speaker":"cuda_fp32"},rows=rows))
    assert r["candidates"]["raw-hard"]["eligible_for_expansion"]
    assert not r["candidates"]["gain"]["eligible_for_expansion"]
    print("Paired assessment selftest passed")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("report",nargs="?",type=Path);p.add_argument("--output",type=Path)
    p.add_argument("--selftest",action="store_true")
    a=p.parse_args()
    if a.selftest:selftest()
    else:
        if a.report is None:p.error("report required")
        s=json.dumps(assess(json.loads(a.report.read_text())),allow_nan=False,indent=2)
        if a.output:a.output.write_text(s)
        else:print(s)
