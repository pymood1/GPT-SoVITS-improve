"""Package paired listening clips with private randomized answer key.
Requires completed paired pilot output (not public audio). No model inference.
"""
import argparse,json,random,shutil
from pathlib import Path
ARMS=("raw","raw-hard","gain","gain-hard","v2ProPlus","real","true-mel-vocoder","oracle-s2")

def package(source,out,seed):
    if out.exists():raise FileExistsError(out)
    rng=random.Random(seed)
    cases=[(ri,ti) for ri in (1,2) for ti in range((ri-1)*2+1,ri*2+1)]
    plan=[]
    for ri,ti in cases:
        arms=list(ARMS);rng.shuffle(arms)
        files=[source/f"ref{ri:02}-text{ti:02}-{arm}-level.wav" for arm in arms]
        if not all(p.is_file() and p.stat().st_size>44 for p in files):raise ValueError("Incomplete listening set")
        plan.append((ri,ti,arms,files))
    out.mkdir();listener=out/"listener";listener.mkdir()
    key=[];ratings=[]
    for i,(ri,ti,arms,files) in enumerate(plan,1):
        folder=listener/f"case-{i:02}";folder.mkdir()
        for j,(arm,path) in enumerate(zip(arms,files),1):
            name=f"clip-{j:02}.wav";shutil.copyfile(path,folder/name)
            key.append(dict(case=i,clip=j,arm=arm,reference_index=ri,text_index=ti))
            ratings.append(dict(case=i,clip=j,artifact_severity=None,naturalness=None,speaker_similarity=None,notes=""))
    (listener/"ratings.json").write_text(json.dumps(ratings,indent=2))
    (listener/"README.txt").write_text("Listen at fixed comfortable volume. Within each case, compare the same sentence. Rate artifact severity 0 (none) to 4 (severe), naturalness 1-5, and speaker similarity 1-5 if a reference is provided; leave missing values null. Do not inspect the answer key. Some clips are diagnostic controls with access to target audio. This is an informal blinded comparison, not a standardized MOS/MUSHRA test.\n")
    # Speaker similarity can be rated against the prompt; label is not a model identity.
    manifest=json.loads((source/"manifest.json").read_text())
    for i,(ri,ti,_,_) in enumerate(plan,1):
        reference=source/manifest["pairs"][ri-1]["reference"]["path"]
        shutil.copyfile(reference,listener/f"case-{i:02}"/"speaker-reference.wav")
    (out/"private-key.json").write_text(json.dumps({"seed":seed,"key":key},indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--seed",type=int,default=72913)
    a=p.parse_args();package(a.source,a.output,a.seed)
