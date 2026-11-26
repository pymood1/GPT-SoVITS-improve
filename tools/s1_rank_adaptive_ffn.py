"""Research-only rank-adaptive SwiGLU modulation branch.

Keep learned gate/down projections and affine RMS/attention untouched.
Approximate only the multiplicative up projection using a truncated SVD.
No automatic training, no checkpoint modification, no production integration.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class RankAdaptiveSwiFFN(nn.Module):
    def __init__(self,source,rank):
        super().__init__()
        d=source.up.in_features
        w=source.up.out_features
        if type(rank) is not int or not 1<=rank<=min(d,w):
            raise ValueError("Invalid modulation rank")
        if source.up.bias is None:
            raise ValueError("Expected learned modulation bias")
        self.gate=copy.deepcopy(source.gate)
        self.down=copy.deepcopy(source.down)
        self.gate.requires_grad_(False);self.down.requires_grad_(False)
        self.rank=rank
        device=source.up.weight.device
        dtype=source.up.weight.dtype
        # SVD is computed on CPU FP64 once at export. No randomly initialized
        # trainable weights survive construction. Split singular values evenly.
        u,s,vh=torch.linalg.svd(source.up.weight.detach().cpu().double(),full_matrices=False)
        root=s[:rank].sqrt()
        self.in_proj=nn.Linear(d,rank,bias=False,device=device,dtype=dtype)
        self.out_proj=nn.Linear(rank,w,bias=True,device=device,dtype=dtype)
        with torch.no_grad():
            self.in_proj.weight.copy_((root[:,None]*vh[:rank]).to(device=device,dtype=dtype))
            self.out_proj.weight.copy_((u[:,:rank]*root[None,:]).to(device=device,dtype=dtype))
            self.out_proj.bias.copy_(source.up.bias)
        self.discarded_energy=float(s[rank:].square().sum()/s.square().sum().clamp_min(1e-30))

    def forward(self,x):
        return self.down(F.silu(self.gate(x))*self.out_proj(self.in_proj(x)))

    def dense_up_weight(self):
        return self.out_proj.weight@self.in_proj.weight

    def modulation_parameter_count(self):
        return sum(p.numel() for m in (self.in_proj,self.out_proj) for p in m.parameters())


def selftest():
    from tools.train_s1_progressive_ffn import SwiFFN
    torch.manual_seed(417)
    dense=SwiFFN(16,64).eval()
    with torch.no_grad():
        # Construct an exact rank-three modulation matrix and nonzero learned bias.
        dense.up.weight.copy_(torch.randn(64,3)@torch.randn(3,16)*.03)
        dense.up.bias.uniform_(.8,1.2)
    reduced=RankAdaptiveSwiFFN(dense,3).eval()
    x=torch.randn(2,9,16)
    torch.testing.assert_close(reduced.dense_up_weight(),dense.up.weight,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(reduced(x),dense(x),rtol=2e-5,atol=2e-6)
    assert reduced.modulation_parameter_count()==3*(16+64)+64
    assert reduced.discarded_energy<1e-12
    assert not any(p.requires_grad for p in reduced.gate.parameters())
    assert not any(p.requires_grad for p in reduced.down.parameters())
    reduced(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in reduced.in_proj.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in reduced.out_proj.parameters())
    # Generic dense case: full rank retains function; truncation is only approximate.
    dense=SwiFFN(16,64).eval()
    full=RankAdaptiveSwiFFN(dense,16).eval()
    low=RankAdaptiveSwiFFN(dense,4).eval()
    torch.testing.assert_close(full(x),dense(x),rtol=2e-5,atol=2e-6)
    assert low.discarded_energy>full.discarded_energy
    print("Rank-adaptive modulation CPU tests passed")


if __name__=="__main__":
    selftest()
