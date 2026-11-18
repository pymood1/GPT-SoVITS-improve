"""Exercise production orchestration without loading Gradio/models."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_function(name, namespace, prefix_only=False):
    tree = ast.parse((ROOT / "webui.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    if prefix_only:
        block = next(n for n in node.body if isinstance(n, ast.Try))
        cutoff = next(i for i, n in enumerate(block.body)
                      if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "listening_reference"
                              for t in n.targets))
        block.body = block.body[:cutoff]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 "webui.py", "exec"), namespace)
    return namespace[name]


class DevPreprocessErrors(unittest.TestCase):
    def dev_run(self, cancel=False):
        ns = dict(
            Path=Path, _training_process_running=lambda: False,
            dev_load_preset=lambda p: {"training": {"cases": []}},
            DEV_EXPERIMENT_PRESET_PATH="fixture", dev_slug=lambda p: str(p),
            _clean_user_path=lambda p: p, dev_available_gpu_ids=lambda *a: [0],
            set_gpu_numbers="0", default_gpu_numbers="0",
            dev_preprocess_gpu_string=lambda p: "0",
            pretrained_gpt_name={"v5dev": "s1"},
            pretrained_sovits_name={"v5dev": "s2"},
            default_bert_path="bert", default_cnhubert_path="hubert",
            GPT_weight_version2root={"v5dev": "output"},
            dev_output_root=lambda p: p,
            _dev_experiment_status=lambda title, lines: "\n".join(lines),
            preprocess_1abcd_cancel_requested=False,
        )
        def child(*args):
            yield ("### 전처리 일괄 실행 실패\n1A fixture dependency missing",)
            ns["preprocess_1abcd_cancel_requested"] = cancel
        ns["open1abcd"] = child
        fn = load_function("run_dev_one_click_experiment", ns, prefix_only=True)
        return fn("fixture.list", "fixture_wav", "fixture")

    def test_dev_exception_keeps_preprocessing_error(self):
        with self.assertRaisesRegex(RuntimeError, "1A fixture dependency missing"):
            list(self.dev_run())

    def test_cancel_at_generator_end_is_not_failure(self):
        list(self.dev_run(cancel=True))

    def run_preprocess(self, child):
        ns = dict(
            _resolve_repo_path=lambda p: p, _usable_training_list=lambda p: True,
            open1abc=lambda *args: iter(child), process_name_1abc="abc",
            process_info=lambda name, state: state,
            run_auto_emotion_sidecar=lambda *args: iter(()),
        )
        fn = load_function("open1abcd", ns)
        return list(fn(*(["fixture"] * 17)))

    def test_abc_failure_keeps_child_diagnostic(self):
        output = self.run_preprocess([("1A failed: missing fixture BERT",)])
        self.assertIn("missing fixture BERT", output[-1][0])

    def test_empty_abc_never_reports_success(self):
        self.assertIn("실패", self.run_preprocess([])[-1][0])

    def test_emotion_failure_keeps_child_diagnostic(self):
        ns = dict(
            _resolve_repo_path=lambda p: p, _usable_training_list=lambda p: True,
            open1abc=lambda *args: iter([("finish",)]), process_name_1abc="abc",
            process_info=lambda name, state: state,
            run_auto_emotion_sidecar=lambda *args: iter([("SER fixture load failed", {})]),
        )
        fn = load_function("open1abcd", ns)
        self.assertIn("SER fixture load failed", list(fn(*(["fixture"] * 17)))[-1][0])


if __name__ == "__main__":
    unittest.main()
