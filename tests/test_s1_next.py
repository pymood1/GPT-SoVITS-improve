"""CPU structural tests for experimental S1 Next; no private data."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"GPT_SoVITS"))
from AR.models.s1_next import NextConfig,S1Next,SemanticResidualBridge,kd_loss,prefix_mask


class NextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.c=NextConfig(dim=32,depth=2,heads=4,ff_dim=88,phonemes=64,codes=32,bert_dim=16)
        self.m=S1Next(self.c).eval()
        self.x=torch.randint(0,64,(1,5))
        self.b=torch.randn(1,16,5)
        self.y=torch.randint(0,32,(1,7))

    def test_prefix_mask(self):
        mask=prefix_mask(3,2,"cpu")
        self.assertTrue(torch.equal(mask,torch.tensor([
            [1,1,1,0,0],[1,1,1,0,0],[1,1,1,0,0],[1,1,1,1,0],[1,1,1,1,1]],dtype=torch.bool)))

    def test_causal_alignment(self):
        a,h=self.m(self.x,self.b,self.y)
        self.assertEqual(a.shape,(1,8,33))
        self.assertEqual(h.shape,(1,7,32))
        changed=self.y.clone()
        changed[:,4:]=(changed[:,4:]+1)%32
        z,_=self.m(self.x,self.b,changed)
        torch.testing.assert_close(a[:,:5],z[:,:5],atol=1e-6,rtol=1e-5)

    def test_cache_equivalence(self):
        full,full_h=self.m(self.x,self.b,self.y)
        x=self.m.text(self.x)+self.m.bert(self.b.transpose(1,2))
        h,cache=self.m._run(x,0,prefix_mask(5,0,"cpu"))
        torch.testing.assert_close(self.m.head(h[:,-1]),full[:,0],atol=2e-6,rtol=1e-5)
        for i in range(7):
            h,cache=self.m._run(self.m.audio(self.y[:,i:i+1]),5+i,None,cache)
            torch.testing.assert_close(self.m.head(h[:,0]),full[:,i+1],atol=2e-6,rtol=1e-5)
            torch.testing.assert_close(h[:,0],full_h[:,i],atol=2e-6,rtol=1e-5)

    def test_rollout_no_eos_and_hidden_alignment(self):
        ids,h=self.m.generate(self.x,self.b,self.y[:,:2],max_new=8,min_new=3,top_k=1)
        self.assertGreaterEqual(ids.size(1),3)
        self.assertLessEqual(ids.size(1),8)
        self.assertTrue((ids<self.c.codes).all())
        _,full=self.m(self.x,self.b,torch.cat((self.y[:,:2],ids),1))
        torch.testing.assert_close(h,full[:,2:],atol=3e-6,rtol=1e-5)

    def test_kd_backward_teacher_frozen(self):
        self.m.train()
        logits,_=self.m(self.x,self.b,self.y)
        teacher=torch.randn_like(logits,requires_grad=True)
        target=F.pad(self.y,(0,1),value=32)
        loss,ce,kl=kd_loss(logits,teacher,target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.m.parameters()))

    def test_bridge_identity_and_gradient(self):
        m=SemanticResidualBridge(32,48,8)
        h=torch.randn(1,7,32)
        e=torch.randn(1,7,48)
        torch.testing.assert_close(m(e,h),e,atol=0,rtol=0)
        m(e,h).square().mean().backward()
        self.assertTrue(torch.isfinite(m.gate.grad))
        self.assertGreater(abs(float(m.gate.grad)),0)
        with self.assertRaises(ValueError):
            m(e,h[:,:-1])

    def test_partial_teacher_transfer(self):
        layers=[]
        for _ in range(2):
            layers.append(SimpleNamespace(self_attn=nn.MultiheadAttention(32,4,batch_first=True)))
        teacher=SimpleNamespace(ar_text_embedding=SimpleNamespace(word_embeddings=nn.Embedding(64,32)),
            ar_audio_embedding=SimpleNamespace(word_embeddings=nn.Embedding(33,32)),
            bert_proj=nn.Linear(16,32),ar_predict_layer=nn.Linear(32,33,bias=False),
            h=SimpleNamespace(layers=layers))
        count=self.m.initialize_from_teacher(teacher)
        self.assertEqual(count,13)
        torch.testing.assert_close(self.m.blocks[0].qkv.weight,layers[0].self_attn.in_proj_weight)
        torch.testing.assert_close(self.m.head.weight,teacher.ar_predict_layer.weight)


if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main()
