"""Round 14: no-quantization FP32 ASR on saved waveforms, matched across arms."""
import argparse
import json
import time
from pathlib import Path
import torch
from tools.kaggle_quality_listening import character_errors, normalized_text


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--backend",choices=("ct2","transformers"),required=True)
    p.add_argument("--expected-count",type=int,default=12)
    a=p.parse_args()
    request=json.loads(a.source.read_text())
    language=request.get("language","ko")
    assert language in ("ko","en")
    assert len(request["samples"])==a.expected_count and a.expected_count>0
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    started=time.monotonic()
    if a.backend=="ct2":
        import ctranslate2
        from faster_whisper import WhisperModel
        root=Path(__file__).resolve().parents[1]
        local=root/"tools/asr/models/faster-whisper-large-v3"
        assert (local/"model.bin").is_file()
        assert "float32" in ctranslate2.get_supported_compute_types("cuda")
        model=WhisperModel(str(local),device="cuda",compute_type="float32",
                           cpu_threads=2,num_workers=1)
        assert model.model.compute_type=="float32",model.model.compute_type
        def transcribe(path):
            segments,_=model.transcribe(path,language=language,beam_size=5,
                                      condition_on_previous_text=False,vad_filter=False)
            return "".join(s.text for s in segments)
        backend="ct2_cuda_float32"
    else:
        import soundfile as sf
        import numpy as np
        import math
        from scipy.signal import resample_poly
        from transformers import AutoProcessor,WhisperForConditionalGeneration
        name="openai/whisper-large-v3"
        proc=AutoProcessor.from_pretrained(name)
        model=WhisperForConditionalGeneration.from_pretrained(name,torch_dtype=torch.float32,
                  attn_implementation="eager").to("cuda").eval()
        assert all(p.dtype==torch.float32 for p in model.parameters())
        @torch.inference_mode()
        def transcribe(path):
            wave,sr=sf.read(path,dtype="float32",always_2d=True)
            wave=wave.mean(axis=1)
            if sr!=16000:
                g=math.gcd(sr,16000)
                wave=resample_poly(wave,16000//g,sr//g)
            assert len(wave)<=480000,"Long audio requires separate evaluation"
            inputs=proc(np.ascontiguousarray(wave),sampling_rate=16000,
                        return_tensors="pt",return_attention_mask=True)
            features=inputs.input_features.to("cuda",torch.float32)
            mask=inputs.attention_mask.to("cuda")
            ids=model.generate(input_features=features,attention_mask=mask,
                language=language,task="transcribe",num_beams=5,do_sample=False,
                return_timestamps=False)
            return proc.batch_decode(ids,skip_special_tokens=True)[0]
        backend="transformers_cuda_float32"
    results=[]
    for i,item in enumerate(request["samples"]):
        actual=transcribe(item["path"])
        errors,count=character_errors(request["texts"][item["text_index"]-1],actual)
        results.append({"label":item["label"],"reference_index":item["reference_index"],
                      "text_index":item["text_index"],"char_errors":errors,"char_count":count,
                      "_normalized":normalized_text(actual)})
        print(json.dumps({"event":"asr_progress","backend":backend,"completed":i+1,"total":len(request["samples"])}),flush=True)
    result={"backend":backend,"scores":results,"seconds":time.monotonic()-started}
    tmp=a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False))
    tmp.replace(a.output)


if __name__=="__main__":
    main()
