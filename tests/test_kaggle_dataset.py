import tempfile
import unittest
from pathlib import Path

from tools.kaggle_dataset import (
    KaggleDatasetError,
    discover_dataset,
    prepare_kaggle_dataset,
)


class KaggleDatasetTests(unittest.TestCase):
    def _dataset(self, root: Path, name: str = "voice"):
        dataset = root / name
        datas = dataset / "datas"
        datas.mkdir(parents=True)
        (datas / "a.wav").write_bytes(b"fake-wav-a")
        (datas / "b.wav").write_bytes(b"fake-wav-b")
        return dataset, datas

    def test_datas_only_enters_auto_list_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, datas = self._dataset(root)
            runtime = root / "runtime"

            result = prepare_kaggle_dataset(
                dataset,
                kaggle_input=root,
                runtime_root=runtime,
            )

            self.assertEqual(result["mode"], "auto_list")
            self.assertEqual(result["training_list"], "")
            self.assertEqual(
                Path(result["datas_dir"]),
                datas.resolve(),
            )
            self.assertEqual(result["audio_count"], 2)
            self.assertTrue(Path(result["manifest"]).is_file())

    def test_existing_list_is_normalized_to_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, datas = self._dataset(root)
            (dataset / "training.list").write_text(
                "C:/old/a.wav|spk|KO|hello\n"
                "C:/old/b.wav|spk|KO|world\n",
                encoding="utf-8",
            )
            runtime = root / "runtime"

            result = prepare_kaggle_dataset(
                dataset,
                kaggle_input=root,
                runtime_root=runtime,
            )

            self.assertEqual(result["mode"], "provided_list")
            training_list = Path(result["training_list"])
            self.assertEqual(
                training_list,
                (runtime / "training.list").resolve(),
            )
            lines = training_list.read_text(
                encoding="utf-8",
            ).splitlines()
            self.assertTrue(lines[0].startswith(str(datas / "a.wav")))
            self.assertTrue(lines[1].startswith(str(datas / "b.wav")))

    def test_explicit_datas_directory_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, datas = self._dataset(root)

            source_root, discovered_datas = discover_dataset(
                datas,
                kaggle_input=root,
                runtime_root=root / "runtime",
            )

            self.assertEqual(source_root, dataset)
            self.assertEqual(discovered_datas, datas)

    def test_ambiguous_scan_requires_explicit_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._dataset(root, "one")
            self._dataset(root, "two")

            with self.assertRaises(KaggleDatasetError):
                discover_dataset(
                    None,
                    kaggle_input=root,
                    runtime_root=root / "runtime",
                )


if __name__ == "__main__":
    unittest.main()
