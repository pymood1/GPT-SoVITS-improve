import json
import sys
import tempfile
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"GPT_SoVITS")]
from tools.evaluate_s1_next_waveforms import read_manifest,token_stats,summarize
from tools.kaggle_s1_next_waveforms import prior_assets,SOURCE,SOURCE_SHA


class WaveformContractTests(unittest.TestCase):
    def test_pinned_prior_assets(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            base=root/SOURCE.split('/')[-1]
            base.mkdir()
            result=base/'s1-next-result.json'
            result.write_text(json.dumps({'code_sha':SOURCE_SHA,'status':'completed'}))
            names=['trajectory_kd/s1-next-best.pt','bridge/aligned-best.pt','bridge/global_mean-best.pt']
            for name in names:
                p=base/'s1-next-pilot'/name
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(b'x'*10000)
            assets=prior_assets(SOURCE,root)
            self.assertEqual(set(assets),{'student','aligned','global_mean'})
            result.write_text(json.dumps({'code_sha':'wrong','status':'completed'}))
            with self.assertRaises(ValueError):
                prior_assets(SOURCE,root)


    def test_independent_reference_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/"a.wav").write_bytes(b"a")
            (root/"b.wav").write_bytes(b"b")
            path=root/"manifest.json"
            m={"references":[{"path":"a.wav","text":"안녕하세요."},
                             {"path":"b.wav","text":"오늘은 맑습니다."}],
               "targets":["다른 문장입니다.","새로운 문장입니다."]}
            path.write_text(json.dumps(m),encoding="utf-8")
            self.assertEqual(read_manifest(path)["references"][0]["path"],str(root/"a.wav"))
            m["targets"][0]="안녕하세요!"
            path.write_text(json.dumps(m),encoding="utf-8")
            with self.assertRaises(ValueError):
                read_manifest(path)
            m["targets"][0]="다른 문장입니다."
            (root/"b.wav").write_bytes(b"a")
            path.write_text(json.dumps(m),encoding="utf-8")
            with self.assertRaises(ValueError):
                read_manifest(path)

    def test_stats_and_failed_denominator(self):
        s=token_stats(torch.tensor([[1,2,2,2,3]]),5)
        self.assertEqual(s,{"tokens":5,"hit_cap":True,"longest_repeat":3})
        rows=[dict(arm="student",status="ok",**s,duration=1.,s1_seconds=.2,total_seconds=.3),
              dict(arm="student",status="failed"),dict(arm="teacher",status="failed")]
        report=summarize(rows)
        self.assertEqual(report["student"]["attempts"],2)
        self.assertEqual(report["student"]["successes"],1)
        self.assertEqual(report["student"]["mean_hit_cap"],1.)
        self.assertEqual(report["teacher"]["successes"],0)

if __name__=="__main__":
    unittest.main()
