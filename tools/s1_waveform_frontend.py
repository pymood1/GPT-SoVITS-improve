"""Verify NLTK in the actual synthesis worker, after GPT-SoVITS frontend imports."""
import os
from pathlib import Path

RESOURCE = "averaged_perceptron_tagger_eng"

def prioritize_paths(existing, paths):
    return list(dict.fromkeys([str(p) for p in paths]+list(existing)))

def ensure_worker_nltk():
    # Some frontend dependencies change NLTK's search paths during import.
    import nltk
    from text import english
    root=Path(__file__).resolve().parents[1]/"runtime/nltk_data"
    root.mkdir(parents=True,exist_ok=True)
    paths=[root]+[Path(p) for p in os.environ.get("NLTK_DATA","").split(os.pathsep) if p]
    os.environ["NLTK_DATA"]=os.pathsep.join(prioritize_paths([],paths))
    nltk.data.path[:]=prioritize_paths(nltk.data.path,paths)
    try:
        nltk.data.find("taggers/"+RESOURCE+"/")
        nltk.pos_tag(["English","test"])
    except LookupError:
        # Per-process fallback avoids two workers writing the same files.
        fallback=root/("worker-"+str(os.getpid()))
        fallback.mkdir(exist_ok=True)
        if not nltk.download(RESOURCE,download_dir=str(fallback),quiet=True,raise_on_error=True):
            raise RuntimeError("NLTK worker resource download failed")
        nltk.data.path[:]=prioritize_paths(nltk.data.path,[fallback,root])
        nltk.data.find("taggers/"+RESOURCE+"/")
        nltk.pos_tag(["English","test"])
    # Test the actual imported G2P implementation, not only isolated nltk.
    phones=english.g2p("A short English test.")
    if not phones:
        raise RuntimeError("Empty English frontend smoke output")
    print("[preflight] worker English G2P verified",flush=True)

def selftest():
    assert prioritize_paths(["/old","/shared"],["/new","/shared"])==["/new","/shared","/old"]
    assert prioritize_paths([],["/a","/a"])==["/a"]
    print("Worker NLTK search path tests passed")

if __name__=="__main__":
    selftest()
