import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"GPT_SoVITS")]
from AR.models.s1_next import SemanticResidualBridge,NextConfig,S1Next,sample_next
from module.s1_next_bridge import FrozenV5Bridge,paired_losses
from tools.train_s1_next import TeacherCache,trajectory_kd
from tools.train_s1_next_bridge_pilot import condition


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Conv1d(8,8,1)
    def forward(self,x,*args):
        return self.linear(x),None


class Quantizer(nn.Module):
    def decode(self,codes):
        return torch.nn.functional.one_hot(codes[:,0],8).float().transpose(1,2)


class Ref(nn.Module):
    def forward(self,x,mask):
        return x[:,:8].mean(-1,keepdim=True)


class CFM(nn.Module):
    def forward(self,mel,lengths,prompt,fea,grad_ckpt):
        return (fea-mel+torch.randn_like(mel)*.1).square().mean()


class MockV5(nn.Module):
    version="v5dev"
    semantic_frame_rate="25hz"
    def __init__(self):
        super().__init__()
        self.quantizer=Quantizer()
        self.ref_enc=Ref()
        self.enc_p=Encoder()
        self.bridge=nn.Conv1d(8,8,1)
        self.wns1=Encoder()
        self.cfm=CFM()


class EfficiencyTests(unittest.TestCase):
    def test_sampling_penalty_and_eos(self):
        logits=torch.tensor([[2.,1.8,-1.,9.]])
        prior=torch.tensor([[0,0]])
        self.assertEqual(int(sample_next(logits,prior,eos=3,allow_eos=False,top_k=1)),0)
        self.assertEqual(int(sample_next(logits,prior,eos=3,allow_eos=False,top_k=1,repetition_penalty=1.35)),1)
        self.assertEqual(int(sample_next(logits,prior,eos=3,allow_eos=True,top_k=1)),3)
        self.assertEqual(float(logits[0,3]),9.)

    def test_trajectory_kd_gradients_and_mode(self):
        cfg=NextConfig(dim=16,depth=1,heads=2,ff_dim=32,phonemes=12,codes=8,bert_dim=4)
        student=S1Next(cfg).train()
        x=torch.tensor([[1,2,3]])
        b=torch.randn(1,4,3)
        y=torch.tensor([[1,2,3,4,5,6]])
        def fake(teacher,text,bert,context):
            self.assertEqual(context.shape[1],6)
            return torch.zeros(1,context.shape[1]+1,9),None
        with patch("tools.train_s1_next.teacher_logits",side_effect=fake):
            loss=trajectory_kd(student,None,x,b,y,temperature=2.,seed=11)
        self.assertTrue(student.training)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(student.head.weight.grad.norm()),0)

    def test_bridge_conditions_do_not_use_future(self):
        h=torch.arange(30*4,dtype=torch.float32).reshape(1,30,4)
        example={"hidden":h}
        m=condition(example,"prompt_mean",None)["hidden"]
        torch.testing.assert_close(m[:,0],h[:,:25].mean(1))
        torch.testing.assert_close(m[:,0],m[:,-1])
        g=condition(example,"global_mean",torch.ones(1,1,4))["hidden"]
        self.assertTrue((g==1).all())
        self.assertIs(condition(example,"aligned",None),example)

    def test_cache_content_and_teacher_identity(self):
        x=torch.ones(1,2,dtype=torch.long)
        b=torch.zeros(1,4,2)
        y=torch.tensor([[1,2,3]])
        t=SimpleNamespace(EOS=7)
        expected=torch.nn.functional.pad(y,(0,1),value=7)
        logits=torch.randn(1,4,8)
        with tempfile.TemporaryDirectory() as d, patch("tools.train_s1_next.teacher_logits",return_value=(logits,expected)) as call:
            cache=TeacherCache(d,"teacher-a")
            a,_=cache.get(t,x,b,y)
            z,_=cache.get(t,x,b,y)
            torch.testing.assert_close(a,z,atol=0,rtol=0)
            self.assertEqual(call.call_count,1)
            cache.get(t,x,b+1,y)
            TeacherCache(d,"teacher-b").get(t,x,b,y)
            self.assertEqual(call.call_count,3)
            self.assertEqual((cache.hits,cache.misses),(1,2))

    def test_zero_projection_initialization(self):
        torch.manual_seed(17)
        m=SemanticResidualBridge(4,8,4,initialization="zero_projection")
        h=torch.randn(1,5,4)
        e=torch.randn(1,5,8)
        torch.testing.assert_close(m(e,h),e,atol=0,rtol=0)
        m(e,h).square().mean().backward()
        self.assertFalse(m.gate.requires_grad)
        self.assertGreater(float(m.proj[-1].weight.grad.norm()),0)
        self.assertIsNone(m.gate.grad)

    def test_frozen_gradient_and_paired_zero_parity(self):
        torch.manual_seed(13)
        s2=MockV5()
        m=FrozenV5Bridge(s2,SemanticResidualBridge(4,8,4)).train()
        example=dict(codes=torch.randint(0,8,(1,1,5)),hidden=torch.randn(1,5,4,requires_grad=True),
                     text=torch.ones(1,4,dtype=torch.long),refer=torch.randn(1,704,12),
                     mel=torch.randn(1,8,20),prompt_frames=4)
        before=torch.random.get_rng_state().clone()
        scores=paired_losses(m,example,23)
        self.assertEqual(scores["hard"],scores["bridge"])
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        loss=m.loss(**example)
        loss.backward()
        self.assertFalse(s2.training)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in s2.parameters()))
        self.assertIsNone(example["hidden"].grad)
        self.assertGreater(abs(float(m.adapter.gate.grad)),0)
        m.adapter.gate.data.fill_(.1)
        m.zero_grad()
        m.loss(**example).backward()
        self.assertGreater(float(m.adapter.proj[-1].weight.grad.norm()),0)

if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main()
