"""Bounded VCTK v0.92 48kHz mic2 paired acquisition. No resampling.
Original corpus CC BY 4.0, Yamagishi et al., doi:10.7488/ds/2645.
"""
import hashlib,io,json
from pathlib import Path
from tools.prepare_acoustic_zeroth_pairs import select,normalize
from tools.acoustic_native_bandwidth import audit

def prepare(output,speakers=2,targets=2,*,revision=None,excluded_speakers=(),excluded_texts=(),excluded_hashes=()):
    import numpy as np
    import soundfile as sf
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi,hf_hub_download
    repo="thelarryproject/vctk";api=HfApi()
    revision=revision or api.dataset_info(repo).sha
    blocked_texts={normalize(t) for t in excluded_texts}
    files=sorted(p for p in api.list_repo_files(repo,repo_type="dataset",revision=revision)
                 if p.startswith("train/") and p.endswith(".parquet"))
    if not files:raise FileNotFoundError("VCTK shards unavailable")
    rows=[];pairs=None;used=[]
    for file in files[:4]:
        local=hf_hub_download(repo,file,repo_type="dataset",revision=revision)
        used.append(file)
        for batch in pq.ParquetFile(local).iter_batches(batch_size=16,columns=["id","speaker_id","mic_id","audio","text_original","text_normalized","duration_s","ctc_mismatch"]):
            for r in batch.to_pylist():
                if r["speaker_id"] in excluded_speakers or r["mic_id"]!="mic2" or r["speaker_id"] in ("p280","p315") or not 3<=r["duration_s"]<=10 or r["ctc_mismatch"] is None or r["ctc_mismatch"]>.3:continue
                raw=r["audio"]["bytes"]
                if not raw:continue
                info=sf.info(io.BytesIO(raw))
                if info.samplerate!=48000 or info.channels!=1 or info.format!="FLAC":continue
                x,sr=sf.read(io.BytesIO(raw),dtype="float32")
                if not np.isfinite(x).all() or not 3<=len(x)/sr<=10:continue
                quality=audit(x,sr)
                if quality["rms_dbfs"]< -45 or quality["clipped_fraction"]>.001:continue
                text=r["text_normalized"].strip()
                if not text or normalize(text) in blocked_texts or hashlib.sha256(x.tobytes()).hexdigest() in excluded_hashes:continue
                rows.append(dict(id=r["id"],speaker_id=r["speaker_id"],text=text,
                    seconds=len(x)/sr,sha256=hashlib.sha256(x.tobytes()).hexdigest(),wave=x,
                    source_sha256=hashlib.sha256(raw).hexdigest(),quality=quality,
                    mic_id=r["mic_id"],ctc_mismatch=r["ctc_mismatch"],original_text=r["text_original"]))
            try:pairs=select(rows,speakers,targets)
            except ValueError:continue
            break
        if pairs is not None:break
    if pairs is None:raise ValueError("Insufficient qualified 48kHz pairs within four shards")
    output.mkdir(parents=True,exist_ok=False)
    for i,pair in enumerate(pairs,1):
        for role,r in [("ref",pair["reference"])]+[(f"target-{j:02}",r) for j,r in enumerate(pair["targets"],1)]:
            wav=output/f"speaker-{i:02}-{role}.wav"
            x=r.pop("wave");sf.write(wav,x,48000,subtype="PCM_16")
            y,sr=sf.read(wav,dtype="float32")
            assert sr==48000 and np.array_equal(x,y),"Source PCM changed"
            r["path"]=wav.name;r["pcm_sha256"]=hashlib.sha256(wav.read_bytes()).hexdigest()
    manifest=dict(dataset=repo,revision=revision,split="train",license="CC-BY-4.0",
        attribution="CSTR VCTK Corpus v0.92, University of Edinburgh, doi:10.7488/ds/2645; thelarryproject",
        sample_rate=48000,verified_same_speaker=True,language="en",mic_id="mic2",
        source_shards=used,pairs=pairs,pretraining_overlap="unknown")
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(dict(speakers=len(pairs),sample_rate=48000,revision=revision,mic="mic2",shards=len(used))))
