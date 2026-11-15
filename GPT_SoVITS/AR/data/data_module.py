# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/data/data_module.py
# reference: https://github.com/lifeiteng/vall-e
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from AR.data.bucket_sampler import DistributedBucketSampler
from AR.data.dataset import Text2SemanticDataset


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
        adaptive = bool(
            self.config["train"].get("adaptive_training", False)
        )
        val_fraction = float(
            self.config["train"].get(
                "adaptive_validation_fraction",
                0.10,
            )
        )
        split_seed = int(
            self.config["train"].get(
                "adaptive_split_seed",
                233333,
            )
        )

        self._train_dataset = Text2SemanticDataset(
            phoneme_path=self.train_phoneme_path,
            semantic_path=self.train_semantic_path,
            max_sec=self.config["data"]["max_sec"],
            pad_val=self.config["data"]["pad_val"],
            adaptive_validation=adaptive,
            validation_fraction=val_fraction,
            split_role="train",
            split_seed=split_seed,
        )
        if adaptive:
            try:
                self._dev_dataset = Text2SemanticDataset(
                    phoneme_path=self.train_phoneme_path,
                    semantic_path=self.train_semantic_path,
                    max_sec=self.config["data"]["max_sec"],
                    pad_val=self.config["data"]["pad_val"],
                    adaptive_validation=True,
                    validation_fraction=val_fraction,
                    split_role="val",
                    split_seed=split_seed,
                )
            except RuntimeError as exc:
                print(
                    "[S1 adaptive] validation disabled: "
                    f"{exc}"
                )
                self.config["train"]["adaptive_training"] = False
                # The first train dataset was already created with holdout
                # names removed. Rebuild it with every usable sample so a
                # failed validation split never costs few-shot speaker data.
                self._train_dataset = Text2SemanticDataset(
                    phoneme_path=self.train_phoneme_path,
                    semantic_path=self.train_semantic_path,
                    max_sec=self.config["data"]["max_sec"],
                    pad_val=self.config["data"]["pad_val"],
                    adaptive_validation=False,
                    split_role="train",
                    split_seed=split_seed,
                )
                self._dev_dataset = self._train_dataset
        else:
            self._dev_dataset = self._train_dataset
        # self._dev_dataset = Text2SemanticDataset(
        #     phoneme_path=self.dev_phoneme_path,
        #     semantic_path=self.dev_semantic_path,
        #     max_sample=self.config['data']['max_eval_sample'],
        #     max_sec=self.config['data']['max_sec'],
        #     pad_val=self.config['data']['pad_val'])

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
        adaptive = bool(
            self.config["train"].get(
                "adaptive_training",
                False,
            )
        )
        if not adaptive:
            # Trainer may already have been constructed with validation
            # enabled before setup discovers that the few-shot dataset is too
            # small for a trustworthy holdout. Return a harmless loader
            # instead of None for Lightning-version compatibility;
            # validation_step exits immediately while the adaptive callback
            # stops at the requested minimum epoch.
            return DataLoader(
                self._dev_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=self._dev_dataset.collate,
                num_workers=0,
            )

        workers = max(1, min(self.num_workers, 4))
        return DataLoader(
            self._dev_dataset,
            batch_size=min(4, len(self._dev_dataset)),
            shuffle=False,
            collate_fn=self._dev_dataset.collate,
            num_workers=workers,
            persistent_workers=workers > 0,
            prefetch_factor=2 if workers > 0 else None,
        )

    # 这个会使用到嘛？
    def test_dataloader(self):
        return DataLoader(
            self._dev_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self._train_dataset.collate,
        )
