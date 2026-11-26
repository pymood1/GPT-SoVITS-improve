"""Experimental frozen V5dev path with a differentiable semantic residual.

Production model methods are untouched. Train only the adapter, not S1/S2.
Inputs must be aligned real examples: hard tokens and post-token S1 states at
25 Hz, original phonemes, reference spectrogram and matching target mel.
"""
import torch
from torch import nn
from torch.nn import functional as F


class FrozenV5Bridge(nn.Module):
    def __init__(self, s2, adapter):
        super().__init__()
        if s2.version != "v5dev" or s2.semantic_frame_rate != "25hz":
            raise ValueError("Only V5dev 25 Hz is supported")
        self.s2=s2.eval().requires_grad_(False)
        self.adapter=adapter

    def train(self, mode=True):
        super().train(mode)
        self.s2.eval()
        return self

    def features(self,codes,hidden,text,refer,*,enabled=True,residual_scale=1.):
        if codes.ndim!=3 or codes.shape[:2]!=(1,1) or hidden.shape[:2]!=(1,codes.size(-1)):
            raise ValueError("Use one aligned utterance with [1,1,T] codes and [1,T,D] states")
        if text.ndim!=2 or text.size(0)!=1 or text.size(1)==0 or refer.ndim!=3 or refer.size(0)!=1 or refer.size(1)<704:
            raise ValueError("Invalid text/reference features")
        if codes.numel()==0 or codes.min()<0 or codes.max()>=1024:
            raise ValueError("Invalid codebook IDs")
        # Only immutable inputs are detached; DO NOT use decode_encp's no_grad decorator.
        with torch.no_grad():
            e=self.s2.quantizer.decode(codes).detach()
            mask=refer.new_ones(1,1,refer.size(-1))
            ge=self.s2.ref_enc(refer[:,:704]*mask,mask).detach()
        if not 0<=residual_scale<=1:
            raise ValueError('Residual scale must be in [0,1]')
        if enabled and residual_scale:
            original=e.transpose(1,2)
            adapted=self.adapter(original,hidden.detach())
            e=(adapted if residual_scale==1 else original+residual_scale*(adapted-original)).transpose(1,2)
        q=F.interpolate(e,scale_factor=2,mode="nearest")
        def length(n):
            return torch.tensor([n],dtype=torch.long,device=q.device)
        x,*_=self.s2.enc_p(q,length(q.size(-1)),text,length(text.size(-1)),ge,1)
        fea=F.interpolate(self.s2.bridge(x),scale_factor=2,mode="nearest")
        fea,_=self.s2.wns1(fea,length(fea.size(-1)),ge)
        return fea,ge

    def loss(self,codes,hidden,text,refer,mel,*,prompt_frames,enabled=True,grad_ckpt=True):
        fea,_=self.features(codes,hidden,text,refer,enabled=enabled)
        n=min(mel.size(-1),fea.size(-1))
        if not 0<=prompt_frames<n:
            raise ValueError("No target mel frames")
        lengths=torch.tensor([n],device=fea.device)
        prompt=torch.tensor([prompt_frames],device=fea.device)
        return self.s2.cfm(mel[...,:n],lengths,prompt,fea[...,:n],grad_ckpt)


def paired_losses(model,example,seed):
    """Common random numbers for paired CFM evaluations; no global RNG side effects."""
    device=example["mel"].device
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=="cuda" else []
    values={}
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        for name,enabled in (("hard",False),("bridge",True)):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(seed)
            values[name]=float(model.loss(**example,enabled=enabled,grad_ckpt=False))
    return values
