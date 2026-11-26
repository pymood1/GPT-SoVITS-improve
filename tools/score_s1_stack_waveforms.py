"""Matched within-run GPU scoring. No CPU fallback or expensive cross-device calibration.
All arms use the identical GPU backend; scores are not claimed to be CPU-equivalent.
"""
import json
import sys
import time
from pathlib import Path
import torch
from tools.kaggle_quality_score import make_scorer

if __name__=="__main__":
    source,stage,target=sys.argv[1:]
    assert stage in ("asr","speaker")
    request=json.loads(Path(source).read_text())
    torch.set_num_threads(request["cpu_threads"])
    started=time.monotonic()
    score=make_scorer(stage,"cuda",request)
    scores=[]
    for i,sample in enumerate(request["samples"]):
        r=score(sample)
        r.pop("_normalized",None)
        scores.append(r)
        print(json.dumps({"stage":stage,"completed":i+1,"total":len(request["samples"])}),flush=True)
    result={"scores":scores,"backend":"cuda_int8_float32" if stage=="asr" else "cuda_fp32",
            "seconds":time.monotonic()-started}
    path=Path(target)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result,allow_nan=False))
    tmp.replace(path)
