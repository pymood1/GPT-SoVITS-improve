"""Prepare verified same-speaker prompt/target pairs from Zeroth-Korean test.
No synthesis, no training. Keep audio and original IDs in private output.
Dataset CC BY 4.0, Lucas Jo and Wonkyum Lee; https://www.openslr.org/40/.
"""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import unicodedata

def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC",text).casefold() if c.isalnum())

def select(rows,speakers=4,targets=2):
    """Stable order, globally distinct text/audio, no prompt reused as target."""
    groups=defaultdict(list)
    for r in rows:
        groups[str(r["speaker_id"])].append(r)
    seen_text=set();seen_hash=set();pairs=[]
    for speaker,items in sorted(groups.items()):
        chosen=[];local_text=set();local_hash=set()
        for r in sorted(items,key=lambda x:str(x["id"])):
            text=normalize(r["text"]); digest=r["sha256"]
            if not text or text in seen_text or text in local_text or digest in seen_hash or digest in local_hash:
                continue
            if not 3<=r["seconds"]<=10:continue
            chosen.append(r);local_text.add(text);local_hash.add(digest)
            if len(chosen)==targets+1:break
        if len(chosen)!=targets+1:continue
        seen_text.update(local_text);seen_hash.update(local_hash)
        pairs.append({"speaker_id":speaker,"reference":chosen[0],"targets":chosen[1:]})
        if len(pairs)==speakers:break
    if len(pairs)!=speakers:raise ValueError("Insufficient eligible distinct speakers and utterances")
    return pairs

def prepare(output,speakers,targets):
    import numpy as np
    import soundfile as sf
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi,hf_hub_download
    repo="kresnik/zeroth_korean"
    api=HfApi()
    info=api.dataset_info(repo)
    revision=info.sha
    paths=[p for p in api.list_repo_files(repo,repo_type="dataset",revision=revision)
           if p.endswith(".parquet") and "test" in p.rsplit("/",1)[-1]]
    if not paths:raise FileNotFoundError("Pinned test parquet unavailable")
    rows=[]
    for path in sorted(paths):
        local=hf_hub_download(repo,path,repo_type="dataset",revision=revision)
        parquet=pq.ParquetFile(local)
        for batch in parquet.iter_batches(batch_size=16,columns=["id","speaker_id","text","audio"]):
            for r in batch.to_pylist():
                raw=r["audio"]["bytes"]
                if not raw:continue
                x,sr=sf.read(io.BytesIO(raw),dtype="float32")
                if sr!=16000 or x.ndim!=1 or not np.isfinite(x).all() or not 3<=len(x)/sr<=10:continue
                rms=float(np.sqrt(np.mean(x.astype(np.float64)**2)))
                if rms<1e-4 or np.mean(np.abs(x)>=.999)>.001:continue
                rows.append(dict(id=str(r["id"]),speaker_id=str(r["speaker_id"]),text=r["text"],
                    seconds=len(x)/sr,sha256=hashlib.sha256(x.tobytes()).hexdigest(),wave=x))
    pairs=select(rows,speakers,targets)
    output.mkdir(parents=True,exist_ok=False)
    for si,pair in enumerate(pairs,1):
        for role,r in [("ref",pair["reference"])]+[(f"target-{j:02}",r) for j,r in enumerate(pair["targets"],1)]:
            wav=output/f"speaker-{si:02}-{role}.wav"
            sf.write(wav,r.pop("wave"),16000,subtype="PCM_16")
            r["path"]=wav.name
            r["pcm_sha256"]=hashlib.sha256(wav.read_bytes()).hexdigest()
            assert str(r["speaker_id"])==pair["speaker_id"]
    manifest=dict(dataset=repo,revision=revision,split="test",license="CC-BY-4.0",
        attribution="Zeroth Project, Lucas Jo and Wonkyum Lee, OpenSLR SLR40",
        sample_rate=16000,verified_same_speaker=True,pairs=pairs)
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(dict(speakers=len(pairs),targets=sum(len(p["targets"]) for p in pairs),
                          revision=revision,sample_rate=16000,split="test")))

def selftest():
    rows=[dict(id=f"{s}-{i}",speaker_id=s,text=f"utterance {s} {i}",sha256=f"{s}-{i}",seconds=5)
          for s in range(5) for i in range(4)]
    # Deliberate cross-speaker duplicate.
    rows[4]["sha256"]=rows[0]["sha256"]
    result=select(rows,4,2)
    allrows=[r for p in result for r in [p["reference"],*p["targets"]]]
    assert len(allrows)==len({r["sha256"] for r in allrows})==12
    assert all(str(r["speaker_id"])==p["speaker_id"] for p in result for r in [p["reference"],*p["targets"]])
    try:select(rows,6,2)
    except ValueError:pass
    else:raise AssertionError("Insufficient-speaker gate failed")
    print("Paired corpus selection selftest passed")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path)
    p.add_argument("--speakers",type=int,default=4)
    p.add_argument("--targets",type=int,default=2)
    p.add_argument("--selftest",action="store_true")
    a=p.parse_args()
    if a.selftest:selftest()
    else:
        if a.output is None or not 2<=a.speakers<=10 or not 1<=a.targets<=4:p.error("Invalid bounded corpus request")
        prepare(a.output,a.speakers,a.targets)
