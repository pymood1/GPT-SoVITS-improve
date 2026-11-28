"""Predeclared semantic robustness gate; exploratory and never a rollout gate."""
import math

KINDS=("clean","noise10","band8","room")
def assess(reports):
    assert len(reports)==2 and len({r["seed"] for r in reports})==2
    output=[]
    for r in reports:
        assert r["status"]=="completed" and r["training"]["updates"]==240
        n=r["test_utterances"];assert n>=12 and r["test_speakers"]>=4 and r["train_speakers"]>=6
        rows=r["rows"];idx={(q["index"],q["kind"],q["arm"]):q for q in rows}
        assert len(rows)==len(idx)==n*len(KINDS)*2
        s={}
        for k in KINDS:
            aa=[idx[i,k,"baseline"] for i in range(n)]
            bb=[idx[i,k,"adapter"] for i in range(n)]
            assert all(a["reference_tokens"]==b["reference_tokens"] and a["reference"]==b["reference"] for a,b in zip(aa,bb))
            count=sum(a["reference_tokens"] for a in aa)
            base=sum(a["unit_errors"] for a in aa)/count
            new=sum(b["unit_errors"] for b in bb)/count
            assert math.isfinite(base) and math.isfinite(new)
            s[k]={"base_ued":base,"adapted_ued":new,"delta_ued":new-base,
                  "improved_utterances":sum(b["unit_errors"]<a["unit_errors"] for a,b in zip(aa,bb)),
                  "worse_utterances":sum(b["unit_errors"]>a["unit_errors"] for a,b in zip(aa,bb)),
                  "min_perplexity_ratio":min(b["candidate"]["perplexity"]/max(a["candidate"]["perplexity"],1e-6) for a,b in zip(aa,bb)),
                  "max_repeat_increase":max(b["candidate"]["longest_repeat"]-a["candidate"]["longest_repeat"] for a,b in zip(aa,bb))}
        assert s["clean"]["base_ued"]==0
        gate={"clean_drift_le_2pct":s["clean"]["adapted_ued"]<=.02,
              "seen_noise_improves":s["noise10"]["delta_ued"]<0 and s["band8"]["delta_ued"]<0,
              "unseen_room_nonregression":s["room"]["delta_ued"]<=0,
              "no_large_perplexity_drop":all(q["min_perplexity_ratio"]>=.8 for q in s.values()),
              "no_large_repeat_growth":all(q["max_repeat_increase"]<=4 for q in s.values())}
        output.append({"seed":r["seed"],"conditions":s,"gate":gate,"passed":all(gate.values())})
    return {"replications":output,"pass_both":all(r["passed"] for r in output),
            "interpretation":"Exploratory token stability gate only. No downstream or perceptual quality claim."}
