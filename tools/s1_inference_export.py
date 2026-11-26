"""Opt-in inference-only export for the frozen round-15 S1.

No checkpoint format change. No training or production integration.
Native RMS and packed gate/up are independent switches for a future benchmark.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class NativeAffineRMS(nn.Module):
    def __init__(self, source):
        super().__init__()
        if source.alpha != 1.:
            raise ValueError("Only fully converted RMS norms can be exported")
        self.register_buffer("weight", source.weight.detach().clone())
        self.register_buffer("bias", source.bias.detach().clone())
        self.normalized_shape=tuple(source.normalized_shape)
        self.eps=float(source.eps)

    def forward(self,x):
        # Keep epsilon and affine bias. Native RMSNorm alone is not equivalent.
        return F.rms_norm(x,self.normalized_shape,self.weight,self.eps)+self.bias


class PackedSwiFFN(nn.Module):
    def __init__(self,source):
        super().__init__()
        self.width=source.gate.out_features
        if source.up.out_features!=self.width or source.gate.in_features!=source.up.in_features:
            raise ValueError("Incompatible gate/up dimensions")
        if source.gate.bias is None or source.up.bias is None:
            raise ValueError("Expected biased checkpoint")
        self.register_buffer("weight",torch.cat((source.gate.weight.detach(),source.up.weight.detach()),0))
        self.register_buffer("bias",torch.cat((source.gate.bias.detach(),source.up.bias.detach()),0))
        self.down=copy.deepcopy(source.down).eval().requires_grad_(False)

    def forward(self,x):
        gate,up=F.linear(x,self.weight,self.bias).split(self.width,dim=-1)
        return self.down(F.silu(gate)*up)


def export_generator(source, *, native_rms=False, packed_ffn=False):
    """Share original frozen teacher; convert only requested inference components.

    This generator intentionally supports cached decoding only. Never swap it into
    the original training or full-forward comparison path.
    """
    from tools.train_s1_rms_polish import Generator, RMSCache
    from AR.models.s1_next import sample_next

    class ExportedGenerator(Generator):
        @torch.inference_mode()
        def generate(self,x,b,prompt,*,max_new,min_new,top_k,repetition_penalty,generator,cached=True):
            if not cached:
                raise ValueError("Use original checkpoint for full-forward reference")
            y=prompt
            cache=RMSCache(self.teacher,self.mods,self.norms,x,b,prompt,max_new)
            for i in range(max_new):
                token=sample_next(cache.z,y,eos=self.teacher.EOS,allow_eos=i>=min_new,
                                  top_k=top_k,repetition_penalty=repetition_penalty,generator=generator)
                if token.item()==self.teacher.EOS:
                    break
                y=torch.cat((y,token),1)
                if i+1<max_new:
                    cache.step(token)
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
            return y[:,prompt.size(1):],None

    if source.norms is None:
        raise ValueError("Expected trained RMS model")
    mods={i:PackedSwiFFN(m) if packed_ffn else m for i,m in source.mods.items()}
    g=ExportedGenerator(source.teacher,mods)
    g.norms=[NativeAffineRMS(n) if native_rms else n for n in source.norms]
    return g


def selftest():
    from types import SimpleNamespace
    from tools.train_s1_affine_rms import AffineRMS
    from tools.train_s1_progressive_ffn import SwiFFN
    from tools.train_s1_rms_polish import Generator,RMSCache
    torch.manual_seed(416)
    n=AffineRMS(nn.LayerNorm(32,eps=1e-5))
    n.alpha=1.
    with torch.no_grad():
        n.weight.uniform_(.2,1.8);n.bias.normal_(0,.1)
    native=NativeAffineRMS(n)
    f=SwiFFN(32,128).eval()
    packed=PackedSwiFFN(f)
    for shape in ((1,1,32),(1,19,32),(3,7,32)):
        for scale in (0.,.001,1.,10.):
            x=torch.randn(*shape)*scale
            torch.testing.assert_close(n(x),native(x),rtol=2e-5,atol=2e-6)
            torch.testing.assert_close(f(x),packed(x),rtol=2e-5,atol=2e-6)
    assert packed.weight.shape==(256,32)
    assert packed.weight.numel()+packed.bias.numel()+sum(p.numel() for p in packed.down.parameters())==sum(p.numel() for p in f.parameters())

    class Pos(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("pe",torch.randn(1,128,32)*.1)
            self.alpha=nn.Parameter(torch.ones(1))
            self.x_scale=1.;self.reverse=False;self.dropout=nn.Identity()
        def extend_pe(self,x):
            assert x.size(1)<=self.pe.size(1)
        def forward(self,x):
            return x+self.alpha*self.pe[:,:x.size(1)]
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn=nn.MultiheadAttention(32,4,batch_first=True)
            self.norm1=nn.LayerNorm(32);self.norm2=nn.LayerNorm(32)
            self.linear1=nn.Linear(32,128);self.linear2=nn.Linear(128,32);self.activation=F.relu
    teacher=SimpleNamespace(ar_text_embedding=nn.Embedding(32,32),bert_proj=nn.Linear(8,32),
        ar_text_position=Pos(),ar_audio_embedding=nn.Embedding(32,32),ar_audio_position=Pos(),
        h=SimpleNamespace(layers=[Layer(),Layer()],norm=None),ar_predict_layer=nn.Linear(32,32),EOS=31)
    mods={i:SwiFFN(32,128).eval() for i in range(2)}
    norms=[AffineRMS(v) for layer in teacher.h.layers for v in (layer.norm1,layer.norm2)]
    for v in norms:
        v.alpha=1.
        with torch.no_grad():v.bias.normal_(0,.1)
    source=Generator(teacher,mods);source.norms=norms
    x=torch.randint(32,(1,9));b=torch.randn(1,8,9);p=torch.randint(31,(1,5))
    tokens=torch.randint(31,(1,12))
    with torch.inference_mode():
        for native_flag,packed_flag in ((True,False),(False,True),(True,True)):
            g=export_generator(source,native_rms=native_flag,packed_ffn=packed_flag)
            a=RMSCache(teacher,source.mods,source.norms,x,b,p,12)
            c=RMSCache(teacher,g.mods,g.norms,x,b,p,12)
            for token in tokens.split(1,dim=1):
                torch.testing.assert_close(a.z,c.z,rtol=3e-5,atol=5e-6)
                assert torch.equal(a.z.argmax(-1),c.z.argmax(-1))
                a.step(token);c.step(token)
            def generate(m):
                return m.generate(x,b,p,max_new=12,min_new=10,top_k=15,repetition_penalty=1.35,
                                  generator=torch.Generator().manual_seed(99))[0]
            assert torch.equal(generate(source),generate(g))
    print("S1 inference export: native affine RMS, packed SwiGLU and cache parity CPU tests passed")


if __name__=="__main__":
    selftest()
