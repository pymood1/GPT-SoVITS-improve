# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/data/data_module.py
# reference: https://github.com/lifeiteng/vall-e
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from AR.data.bucket_sampler import DistributedBucketSampler
from AR.data.dataset import Text2SemanticDataset
from tools.grokking import (
    deterministic_holdout_indices,
    repeat_indices_to_minimum,
    resolve_grokking_config,
)


class _DatasetView:
    """Index view that preserves the bucket sampler / collate contract."""

    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

    def get_sample_length(self, idx):
        return self.base_dataset.get_sample_length(self.indices[idx])

    def collate(self, examples):
        return self.base_dataset.collate(examples)


class Text2SemanticDataModule(LightningDataModule):
    def __init__(
        self,
        config,
        train_semantic_path,
        train_phoneme_path,
        dev_semantic_path=None,
        dev_phoneme_path=None,
    ):
        super().__init__()
        self.config = config
        self.train_semantic_path = train_semantic_path
        self.train_phoneme_path = train_phoneme_path
        self.dev_semantic_path = dev_semantic_path
        self.dev_phoneme_path = dev_phoneme_path
        self.num_workers = self.config["data"]["num_workers"]

    def prepare_data(self):
        pass

    def setup(self, stage=None, output_logs=False):
        base_dataset = Text2SemanticDataset(
            phoneme_path=self.train_phoneme_path,
            semantic_path=self.train_semantic_path,
            max_sec=self.config["data"]["max_sec"],
            pad_val=self.config["data"]["pad_val"],
        )

        grokking = resolve_grokking_config(self.config["train"], "s1")
        if not grokking["enabled"]:
            self._train_dataset = base_dataset
            self._dev_dataset = base_dataset
            return

        train_indices, val_indices = deterministic_holdout_indices(
            base_dataset.item_names,
            fraction=float(grokking["validation_fraction"]),
            min_validation_samples=int(grokking["min_validation_samples"]),
            seed=int(grokking["seed"]),
        )
        train_indices = repeat_indices_to_minimum(train_indices, minimum=100)
        self._train_dataset = _DatasetView(base_dataset, train_indices)
        self._dev_dataset = _DatasetView(base_dataset, val_indices)
        print(
            "[grokking:S1] held-out split "
            f"train={len(self._train_dataset)} "
            f"val={len(self._dev_dataset)} "
            f"unique_val={len(val_indices)}"
        )

    def train_dataloader(self):
        batch_size = (
            self.config["train"]["batch_size"] // 2
            if self.config["train"].get("if_dpo", False) is True
            else self.config["train"]["batch_size"]
        )
        batch_size = max(min(batch_size, len(self._train_dataset) // 4), 1)  # 防止不保存
        sampler = DistributedBucketSampler(self._train_dataset, batch_size=batch_size)
        return DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self._train_dataset.collate,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=16,
        )

    def val_dataloader(self):
        grokking = resolve_grokking_config(self.config["train"], "s1")
        workers = self.num_workers if grokking["enabled"] else max(self.num_workers, 12)
        kwargs = {
            "dataset": self._dev_dataset,
            "batch_size": 1,
            "shuffle": False,
            "collate_fn": self._dev_dataset.collate,
            "num_workers": workers,
        }
        if workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4 if grokking["enabled"] else 16
        return DataLoader(**kwargs)

    # 这个会使用到嘛？
    def test_dataloader(self):
        return DataLoader(
            self._dev_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self._train_dataset.collate,
        )
