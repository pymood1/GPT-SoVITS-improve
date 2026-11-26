"""CPU Gloo proof of the global token-weighted DDP update, using real S1 Next."""
import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"GPT_SoVITS")]
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from AR.models.s1_next import S1Next,NextConfig,kd_loss
from tools.train_s1_next_ddp import batches,gradient_multiplier


def example(i):
    g=torch.Generator().manual_seed(2027+i)
    x=torch.randint(0,30,(1,4+i),generator=g)
    b=torch.randn(1,4,4+i,generator=g)
    y=torch.randint(0,12,(1,3+i),generator=g)
    t=torch.nn.functional.pad(y,(0,1),value=12)
    z=torch.randn(1,4+i,13,generator=g)
    return x,b,y,t,z


def worker(rank,init):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method=init,rank=rank,world_size=2)
    try:
        torch.manual_seed(73)
        config=NextConfig(dim=16,depth=1,heads=2,ff_dim=32,phonemes=30,codes=12,bert_dim=4)
        model=S1Next(config)
        ref=S1Next(config)
        ref.load_state_dict(model.state_dict())
        ddp=DDP(model,gradient_as_bucket_view=True,broadcast_buffers=False)
        data=[example(i) for i in range(4)]
        total=sum(e[3].numel() for e in data)
        ids=[rank,rank+2]
        for j,i in enumerate(ids):
            x,b,y,t,z=data[i]
            with (ddp.no_sync() if j==0 else contextlib.nullcontext()):
                logits,_=ddp(x,b,y)
                loss,*_=kd_loss(logits,z,t,alpha=.7)
                (loss*t.numel()*gradient_multiplier(2,total)).backward()
        for x,b,y,t,z in data:
            logits,_=ref(x,b,y)
            loss,*_=kd_loss(logits,z,t,alpha=.7)
            (loss*t.numel()/total).backward()
        for a,b in zip(model.parameters(),ref.parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=1e-4)
    finally:
        dist.destroy_process_group()


class DDPTests(unittest.TestCase):
    def test_shards_and_remainder(self):
        result=list(batches(list(range(13)),2,3))
        self.assertEqual([len(g[0]) for g in result],[3,3])
        flat=[i for g in result for shard in g for i in shard]
        self.assertEqual(sorted(flat),list(range(12)))
        result=list(batches(list(range(10)),2,3))
        self.assertEqual([len(g[0]) for g in result],[3,2])
        self.assertEqual(sorted(i for g in result for shard in g for i in shard),list(range(10)))

    def test_real_distributed_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker,args=("file://"+str(Path(tmp)/"rendezvous"),),nprocs=2,join=True)

if __name__=="__main__":
    unittest.main()
