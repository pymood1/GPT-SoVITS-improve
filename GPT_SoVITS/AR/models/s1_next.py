"""Experimental S1 Next. No changes to the existing inference/training path."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NextConfig:
    dim: int = 512
    depth: int = 24
    heads: int = 16
    ff_dim: int = 1344
    phonemes: int = 732
    codes: int = 1024
    bert_dim: int = 1024
    rope_theta: float = 10000.0

    @classmethod
    def from_teacher(cls, config):
        c = config["model"]
        d = c["hidden_dim"]
        # 3*d*ff_dim ~= 8*d*d, rounded to a multiple of 64.
        return cls(dim=d, depth=c["n_layer"], heads=c["head"],
                   ff_dim=round((8*d/3)/64)*64, phonemes=c["phoneme_vocab_size"],
                   codes=c["vocab_size"]-1)


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        z = x.float()
        return (z * torch.rsqrt(z.square().mean(-1, keepdim=True)+1e-6)).to(x.dtype) * self.weight.to(x.dtype)


def rope_terms(pos, d, theta):
    freq = theta ** (-torch.arange(0,d,2,device=pos.device,dtype=torch.float32)/d)
    a = pos.float()[:,None,:,None]*freq
    return a.cos(), a.sin()


def rotary(x, terms):
    cos,sin=terms
    even,odd=x.float()[...,0::2],x.float()[...,1::2]
    return torch.stack((even*cos-odd*sin,even*sin+odd*cos),-1).flatten(-2).to(x.dtype)


def prefix_mask(text_len, audio_len, device):
    # Boolean SDPA mask: True permits attention. Full text prefix; causal audio.
    n = text_len+audio_len
    i = torch.arange(n, device=device)[:, None]
    j = torch.arange(n, device=device)[None, :]
    return (j < text_len) | ((i >= text_len) & (j <= i))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.theta = c.heads, c.rope_theta
        self.n1, self.n2 = RMSNorm(c.dim), RMSNorm(c.dim)
        self.qkv = nn.Linear(c.dim, 3*c.dim, bias=True)
        self.out = nn.Linear(c.dim, c.dim, bias=True)
        self.gate = nn.Linear(c.dim, c.ff_dim, bias=False)
        self.up = nn.Linear(c.dim, c.ff_dim, bias=False)
        self.down = nn.Linear(c.ff_dim, c.dim, bias=False)

    def forward(self, x, terms, mask, cache=None):
        b, n, d = x.shape
        q,k,v = self.qkv(self.n1(x)).view(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        q,k = rotary(q,terms), rotary(k,terms)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        a = F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        x = x + self.out(a.transpose(1,2).reshape(b,n,d))
        z = self.n2(x)
        x = x + self.down(F.silu(self.gate(z))*self.up(z))
        return x, (k,v)


class S1Next(nn.Module):
    """Unpadded, equal-length batches only. Train on per-example slices initially."""
    def __init__(self, cfg=NextConfig()):
        super().__init__()
        if cfg.dim % cfg.heads or (cfg.dim//cfg.heads)%2 or cfg.ff_dim<1:
            raise ValueError("Invalid head/FFN dimensions")
        self.cfg = cfg
        self.text = nn.Embedding(cfg.phonemes,cfg.dim)
        self.bert = nn.Linear(cfg.bert_dim,cfg.dim)
        self.audio = nn.Embedding(cfg.codes+1,cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.final = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim,cfg.codes+1,bias=False)

    def _run(self, x, start, mask, caches=None):
        pos = torch.arange(start,start+x.shape[1],device=x.device).expand(x.shape[0],-1)
        terms = rope_terms(pos,self.cfg.dim//self.cfg.heads,self.cfg.rope_theta)
        out = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x,terms,mask,None if caches is None else caches[i])
            out.append(kv)
        return self.final(x), out

    def forward(self, text, bert, audio):
        # bert [B,1024,T_text]; audio contains actual previous tokens, no EOS.
        if text.ndim != 2 or text.size(1)==0 or audio.ndim!=2 or bert.shape!=(text.size(0),self.cfg.bert_dim,text.size(1)):
            raise ValueError("Unpadded text/bert/audio expected")
        x = torch.cat((self.text(text)+self.bert(bert.transpose(1,2)),self.audio(audio)),dim=1)
        hidden,_ = self._run(x,0,prefix_mask(text.size(1),audio.size(1),x.device))
        # Last text position predicts first token; last audio position predicts EOS.
        return self.head(hidden[:,text.size(1)-1:]), hidden[:,text.size(1):]

    @torch.no_grad()
    def generate(self, text, bert, prompt, *, max_new=512, min_new=1, temperature=1., top_k=15, generator=None):
        """Single utterance. Return only new IDs and their post-token states, never EOS."""
        if self.training or text.size(0)!=1 or prompt.size(0)!=1 or text.size(1)==0 or temperature<=0 or not 0<=min_new<=max_new:
            raise ValueError("eval mode, single unpadded utterance, valid sampling budget required")
        x = torch.cat((self.text(text)+self.bert(bert.transpose(1,2)),self.audio(prompt)),dim=1)
        h, caches = self._run(x,0,prefix_mask(text.size(1),prompt.size(1),x.device))
        ids, states = [], []
        eos = self.cfg.codes
        position = x.size(1)
        for i in range(max_new):
            logits = self.head(h[:,-1]).float()/temperature
            if i<min_new:
                logits[:,eos] = -torch.inf
            if top_k>0:
                k = min(top_k,logits.size(-1))
                v,ind = logits.topk(k)
                logits = torch.full_like(logits,-torch.inf).scatter(-1,ind,v)
            token = torch.multinomial(logits.softmax(-1),1,generator=generator)
            if token.item()==eos:
                break
            h,caches = self._run(self.audio(token),position,None,caches)
            position += 1
            ids.append(token)
            states.append(h)
        return (torch.cat(ids,dim=1) if ids else prompt[:,:0],
                torch.cat(states,dim=1) if states else x[:,:0])

    @torch.no_grad()
    def initialize_from_teacher(self, teacher):
        # Compatible tensors only: exact embeddings, BERT projection, head, QKVO.
        # ReLU -> SwiGLU / PostLN -> PreRMSNorm are deliberately not copied.
        pairs = [(self.text.weight,teacher.ar_text_embedding.word_embeddings.weight),
                 (self.audio.weight,teacher.ar_audio_embedding.word_embeddings.weight),
                 (self.bert.weight,teacher.bert_proj.weight),(self.bert.bias,teacher.bert_proj.bias),
                 (self.head.weight,teacher.ar_predict_layer.weight)]
        if len(teacher.h.layers)!=len(self.blocks):
            raise ValueError("Teacher depth mismatch")
        for b,t in zip(self.blocks,teacher.h.layers):
            a=t.self_attn
            pairs += [(b.qkv.weight,a.in_proj_weight),(b.qkv.bias,a.in_proj_bias),
                      (b.out.weight,a.out_proj.weight),(b.out.bias,a.out_proj.bias)]
        if any(x.shape!=y.shape for x,y in pairs):
            raise ValueError("Teacher tensor shape mismatch")
        for x,y in pairs:
            x.copy_(y)
        return len(pairs)


def kd_loss(logits, teacher_logits, target, *, temperature=2., alpha=.5):
    if temperature<=0 or not 0<=alpha<=1:
        raise ValueError("Invalid KD parameters")
    if logits.shape!=teacher_logits.shape or logits.shape[:-1]!=target.shape:
        raise ValueError("KD alignment mismatch")
    ce=F.cross_entropy(logits.float().flatten(0,1),target.flatten())
    kl=F.kl_div(F.log_softmax(logits.float()/temperature,-1),
                F.softmax(teacher_logits.detach().float()/temperature,-1),reduction="none").sum(-1).mean()*temperature**2
    return (1-alpha)*ce+alpha*kl,ce.detach(),kl.detach()


class SemanticResidualBridge(nn.Module):
    """Research-only post-token state adapter. Does not alter hard-token baseline."""
    def __init__(self, source_dim=512, target_dim=768, rank=128):
        super().__init__()
        self.proj=nn.Sequential(RMSNorm(source_dim),nn.Linear(source_dim,rank),nn.SiLU(),
                                nn.Linear(rank,target_dim,bias=False))
        self.gate=nn.Parameter(torch.zeros(()))

    def forward(self, hard_embedding, post_token_hidden):
        if hard_embedding.ndim!=3 or post_token_hidden.ndim!=3 or hard_embedding.shape[:2]!=post_token_hidden.shape[:2]:
            raise ValueError("Semantic frame alignment mismatch")
        return hard_embedding + self.gate.tanh()*self.proj(post_token_hidden)
