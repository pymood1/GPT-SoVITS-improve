"""Prepare a bounded, public FLEURS Korean pilot without ASR or private data.

Run in an initialized GPT-SoVITS environment. Dataset: google/fleurs,
CC BY 4.0, pinned parquet revision. Official train/validation are speaker
disjoint per dataset card. Exact duplicate audio and text are excluded.
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"GPT_SoVITS")]
REV="3dfbbb3b3cfb7d48550c24defda9876bbb73f8ac"


def prepare_rows(split,count,root,seen_audio,seen_text,cache,*,min_count=None):
    required=count if min_count is None else min_count
    if not 0<=required<=count or count<1:
        raise ValueError('Invalid corpus count')
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    import soundfile as sf
    import numpy as np
    import unicodedata
    path=hf_hub_download("google/fleurs",f"ko_kr/{split}-00000-of-00001.parquet",
                         repo_type="dataset",revision=REV,cache_dir=str(cache))
    wav=root/"wav"
    wav.mkdir(parents=True,exist_ok=False)
    rows=[]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8,columns=["audio","transcription"]):
        for row in batch.to_pylist():
            text=" ".join(row["transcription"].split()).replace("|"," ")
            key=unicodedata.normalize("NFKC",text).casefold()
            if not key or key in seen_text:
                continue
            raw=row["audio"]["bytes"]
            if not raw:
                raise ValueError("FLEURS audio bytes missing")
            y,sr=sf.read(io.BytesIO(raw),dtype="float32")
            if y.ndim!=1 or not np.isfinite(y).all() or not 3<=len(y)/sr<=10 or np.max(np.abs(y))<1e-3:
                continue
            digest=hashlib.sha256(y.tobytes()+str(sr).encode()).hexdigest()
            if digest in seen_audio:
                continue
            name=f"{split}-{digest[:20]}.wav"
            sf.write(wav/name,y,sr,subtype="PCM_16")
            rows.append((name,text,digest))
            seen_audio.add(digest)
            seen_text.add(key)
            if len(rows)==count:
                break
        if len(rows)==count:
            break
    if len(rows)<required:
        raise RuntimeError("Insufficient eligible FLEURS utterances")
    lst=root/"input.list"
    lst.write_text("".join(f"{wav/name}|fleurs-{split}|ko|{text}\n" for name,text,_ in rows),encoding="utf-8")
    return lst,rows


def preprocess(root,lst,rows,teacher_path,device):
    import torch
    from text.multilingual_bert import checkpoint_multilingual_bert_config
    config=torch.load(teacher_path,map_location="cpu",weights_only=False)["config"]
    multilingual,model=checkpoint_multilingual_bert_config(config)
    pretrained=ROOT/"GPT_SoVITS/pretrained_models"
    env=os.environ.copy()
    env.update(inp_text=str(lst),inp_wav_dir=str(root/"wav"),exp_name=root.name,
               i_part="0",all_parts="1",opt_dir=str(root),_CUDA_VISIBLE_DEVICES=device,
               bert_pretrained_dir=str(pretrained/"chinese-roberta-wwm-ext-large"),
               cnhubert_base_dir=str(pretrained/"chinese-hubert-base"),
               pretrained_s2G=str(pretrained/"gsv-v5-pretrained/s2Gv5dev.pth"),
               s2config_path=str(ROOT/"GPT_SoVITS/configs/s2.json"),
               use_multilingual_bert=str(multilingual),multilingual_bert_model=model,
               is_half="False",version="v2",hz="25hz",
               PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/"GPT_SoVITS")]))
    for script in ("1-get-text.py","2-get-hubert-wav32k.py","3-get-semantic.py"):
        subprocess.run([sys.executable,"-u",str(ROOT/"GPT_SoVITS/prepare_datasets"/script)],
                       cwd=ROOT,env=env,check=True)
    text=(root/"2-name2text-0.txt").read_text(encoding="utf-8").splitlines()
    sem=(root/"6-name2semantic-0.tsv").read_text(encoding="utf-8").splitlines()
    ids={x[0] for x in rows}
    for lines in (text,sem):
        if len(lines)!=len(ids) or {x.split("\t",1)[0] for x in lines}!=ids:
            raise ValueError("Incomplete canonical preprocessing")
    (root/"2-name2text.txt").write_text("\n".join(text)+"\n",encoding="utf-8")
    (root/"6-name2semantic.tsv").write_text("item_name\tsemantic_audio\n"+"\n".join(sem)+"\n",encoding="utf-8")
    (root/"provenance.json").write_text(json.dumps({"dataset":"google/fleurs","revision":REV,
        "license":"CC-BY-4.0","language":"ko_kr","split":root.name,
        "audio_sha256":[x[2] for x in rows],"multilingual_bert":multilingual},indent=2),encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--train-count",type=int,default=64)
    p.add_argument("--val-count",type=int,default=16)
    p.add_argument("--test-count",type=int,default=0)
    p.add_argument("--gpu",default="0")
    args=p.parse_args()
    root=args.root.resolve()
    root.mkdir(parents=True,exist_ok=False)
    audio,text=set(),set()
    splits=[("train",args.train_count),("validation",args.val_count)]
    if args.test_count:
        splits.append(("test",args.test_count))
    for split,count in splits:
        if count<1: p.error("Positive counts required")
        folder=root/split
        lst,rows=prepare_rows(split,count,folder,audio,text,root/"hf-cache")
        preprocess(folder,lst,rows,args.teacher.resolve(),args.gpu)
        print(json.dumps({"split":split,"count":count}),flush=True)

if __name__=="__main__":
    main()
