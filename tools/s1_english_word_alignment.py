"""Experimental English G2P-word/BERT alignment. Does not alter native phones."""
import torch
def word_spans(text,words):
    spans=[];cursor=0
    for word in words:
        i=text.find(word,cursor)
        if i<0 or text[cursor:i].strip():return None
        spans.append((i,i+len(word)));cursor=i+len(word)
    return spans if not text[cursor:].strip() else None

def grouped_phones(raw,words,convert):
    groups=[[]]
    for ph in raw:
        if ph==" ":groups.append([])
        else:groups[-1].append(ph)
    if len(groups)!=len(words):return None
    return [convert(g) for g in groups]

def align_words(hidden,offsets,spans,groups,phones):
    if spans is None or groups is None or len(spans)!=len(groups):return None
    flat=[p for g in groups for p in g]
    if flat!=list(phones):return None
    vectors=[]
    for (a,b),group in zip(spans,groups):
        if not group:continue
        overlap=[max(0,min(b,int(e))-max(a,int(s))) for s,e in offsets]
        w=hidden.new_tensor(overlap)
        if not w.sum()>0:return None
        f=(hidden*w[:,None]).sum(0)/w.sum()
        vectors.extend([f]*len(group))
    return torch.stack(vectors) if vectors else None

def selftest():
    assert word_spans("a cat.",["a","cat","."])==[(0,1),(2,5),(5,6)]
    assert word_spans("a cat",["a","dog"]) is None
    assert grouped_phones(["A"," ","K","AE","T"],["a","cat"],list)==[["A"],["K","AE","T"]]
    hidden=torch.tensor([[1.,0.],[0.,1.],[0.,3.]])
    out=align_words(hidden,[(0,1),(2,3),(3,5)],[(0,1),(2,5)],[[1],[2,3]],[1,2,3])
    torch.testing.assert_close(out,torch.tensor([[1.,0.],[0.,7/3],[0.,7/3]]))
    assert align_words(hidden,[(0,1),(2,3),(3,5)],[(0,1),(2,5)],[[1],[2,3]],[1,3,2]) is None
    print("Word span, overlap-weighted pooling and strict native phone parity tests passed")

@torch.no_grad()
def variants(extractor,text,phones,version):
    from text import english,cleaned_text_to_sequence
    from text.multilingual_bert import align_token_features_to_phones,pad_hidden_to_gpt_dimension
    encoded=extractor.tokenizer(text,return_tensors="pt",return_offsets_mapping=True,truncation=True,max_length=512)
    offsets=encoded.pop("offset_mapping")[0].tolist()
    result=extractor.model(**{k:v.to(extractor.device) for k,v in encoded.items()},output_hidden_states=True,return_dict=True)
    h=result.hidden_states[-3][0].float().cpu()
    valid=torch.tensor([e>s for s,e in offsets])
    assert valid.any()
    nearest=align_token_features_to_phones(h,offsets,text_length=len(text),phone_count=len(phones))
    pooled=h[valid].mean(0).expand(len(phones),-1).clone()
    words=english.word_tokenize(text)
    def convert(raw):
        ph=english.replace_phs([p if p!="<unk>" else "UNK" for p in raw if p not in (" ","<pad>","UW","</s>","<s>")])
        return cleaned_text_to_sequence(ph,version)
    groups=grouped_phones(english._g2p(text),words,convert)
    aligned=align_words(h,offsets,word_spans(text,words),groups,phones)
    exact=aligned is not None
    if not exact:aligned=nearest
    return {k:pad_hidden_to_gpt_dimension(x).T.contiguous() for k,x in (("nearest",nearest),("word",aligned),("pooled",pooled))},exact

if __name__=="__main__":selftest()
