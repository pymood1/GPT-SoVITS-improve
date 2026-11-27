"""Reproduce currently audited training contracts from real source AST.
Diagnostic only: checks known defects, not tests claiming they were fixed.
"""
import ast,json
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]/"GPT_SoVITS"
def extract(path,name):
    tree=ast.parse((ROOT/path).read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    ns=dict(np=np,torch=torch,F=F,List=List)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),ns)
    return ns[name]
def main():
    batch=extract("AR/data/dataset.py","batch_sequences")
    left=extract("AR/models/utils.py","make_pad_mask_left")
    right=extract("AR/models/utils.py","make_pad_mask")
    pad=extract("AR/models/t2s_model.py","pad_y_eos")
    reject=extract("AR/models/utils.py","make_reject_y")
    x=batch([np.array([10,20,30]),np.array([11,12,13,14,15])])
    lengths=torch.tensor([3,5])
    actual=left(lengths);expected=right(lengths)
    assert x[0].tolist()==[10,20,30,0,0]
    assert actual[0].tolist()==[True,True,False,False,False]
    assert expected[0].tolist()==[False,False,False,True,True]
    y=torch.tensor([[10,20,30,1024,1024],[11,12,13,14,15]])
    mask=right(lengths).long()
    inp,tgt=pad(None,y*(1-mask),mask,1024)
    assert tgt[0].tolist()==[10,20,30,1024,1024,1024]
    assert tgt[1].tolist()==[11,12,13,14,15,1024]
    hits=0
    for seed in range(64):
        torch.manual_seed(seed);r,rl=reject(y,lengths)
        hits+=int((r[0,:rl[0]]==1024).any())
    assert hits>0
    print(json.dumps({"right_padded_text":x[0].tolist(),"actual_mask":actual[0].tolist(),
        "expected_mask":expected[0].tolist(),"actual_labels":tgt[0].tolist(),
        "post_end_extra_eos":2,"dpo_rejected_eos_contamination_cases":hits,"dpo_trials":64}),flush=True)
if __name__=="__main__":main()
