# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_lightning_module.py
# reference: https://github.com/lifeiteng/vall-e
import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
from collections import OrderedDict
from typing import Dict

import torch
from pytorch_lightning import LightningModule

from AR.models.t2s_model import Text2SemanticDecoder
from AR.modules.activation import MultiheadAttention
from AR.modules.lr_schedulers import WarmupCosineLRSchedule
from AR.modules.optim import ScaledAdam


class Text2SemanticLightningModule(LightningModule):
    def __init__(self, config, output_dir, is_train=True):
        super().__init__()
        self.config = config
        self.top_k = 3
        self.model = Text2SemanticDecoder(config=config, top_k=self.top_k)
        pretrained_s1 = config.get("pretrained_s1")
        if pretrained_s1 and is_train:
            # print(self.load_state_dict(torch.load(pretrained_s1,map_location="cpu")["state_dict"]))
            print(
                self.load_state_dict(
                    torch.load(
                        pretrained_s1,
                        map_location="cpu",
                        weights_only=False,
                    )["weight"],
                )
            )

        self.s1_adapter_mode = "full"
        if is_train:
            self._configure_s1_adapter(pretrained_s1)
            self.automatic_optimization = False
            self.grad_accum_steps = int(
                config["train"].get("grad_accumulation_steps", 4)
            )
            if self.grad_accum_steps <= 0:
                raise ValueError("grad_accumulation_steps must be positive")
            self._grad_accum_count = 0
            self.save_hyperparameters()
            self.eval_dir = output_dir / "eval"
            self.eval_dir.mkdir(parents=True, exist_ok=True)

    def _configure_s1_adapter(self, pretrained_s1):
        train_config = self.config.get("train", {})
        mode = str(
            train_config.get("s1_adapter", "full") or "full"
        ).strip().lower()
        aliases = {
            "none": "full",
            "full_ft": "full",
            "full-finetune": "full",
            "v_o_lora": "lora_v_o",
            "v+o": "lora_v_o",
            "qkv_o_lora": "lora_qkv_o",
            "qkv+o": "lora_qkv_o",
        }
        mode = aliases.get(mode, mode)
        if mode not in {
            "full",
            "lora_v_o",
            "lora_qkv_o",
        }:
            raise ValueError(
                f"unsupported S1 adapter mode: {mode!r}"
            )
        self.s1_adapter_mode = mode

        if mode == "full":
            return
        if not pretrained_s1:
            raise ValueError(
                "S1 LoRA requires train.pretrained_s1 so the frozen "
                "backbone has a pretrained reference geometry"
            )

        rank = int(train_config.get("s1_lora_rank", 8))
        alpha = float(train_config.get("s1_lora_alpha", 16.0))
        if rank <= 0:
            raise ValueError("train.s1_lora_rank must be positive")
        if alpha <= 0:
            raise ValueError("train.s1_lora_alpha must be positive")

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        target = "v" if mode == "lora_v_o" else "qkv"
        attention_count = 0
        for module in self.model.modules():
            if not isinstance(module, MultiheadAttention):
                continue
            module.enable_lora(
                target=target,
                rank=rank,
                alpha=alpha,
            )
            attention_count += 1

        if attention_count == 0:
            raise RuntimeError(
                "S1 LoRA found no MultiheadAttention modules"
            )

        trainable = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        total = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )
        ratio = 100.0 * trainable / max(total, 1)
        print(
            "[S1 adapter] "
            f"mode={mode} target={target}+out "
            f"rank={rank} alpha={alpha:g} "
            f"attention_layers={attention_count} "
            f"trainable={trainable:,}/{total:,} "
            f"({ratio:.3f}%)"
        )

    def inference_state_dict(self):
        """Return a normal S1 state_dict with training LoRA merged.

        Full Lightning resume checkpoints keep the explicit LoRA parameters.
        Compact GPT checkpoints call this method so inference/loaders remain
        completely LoRA-unaware and backward compatible.
        """
        state = self.state_dict()
        if self.s1_adapter_mode == "full":
            return state

        exported = OrderedDict(
            (key, value)
            for key, value in state.items()
            if ".lora_" not in key
        )
        merged = 0
        for name, module in self.model.named_modules():
            if (
                not isinstance(module, MultiheadAttention)
                or not module.lora_enabled
            ):
                continue
            prefix = f"model.{name}."
            in_key = prefix + "in_proj_weight"
            out_key = prefix + "out_proj.weight"
            if in_key not in exported or out_key not in exported:
                raise RuntimeError(
                    "S1 LoRA compact export could not find base "
                    f"attention weights for {name!r}"
                )
            exported[in_key] = (
                module.merged_in_proj_weight().detach()
            )
            exported[out_key] = (
                module.merged_out_proj_weight().detach()
            )
            merged += 1

        if merged == 0:
            raise RuntimeError(
                "S1 LoRA compact export found no active adapters"
            )
        return exported

    def training_step(self, batch: Dict, batch_idx: int):
        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        forward = self.model.forward if self.config["train"].get("if_dpo", False) == True else self.model.forward_old
        loss, acc = forward(
            batch["phoneme_ids"],
            batch["phoneme_ids_len"],
            batch["semantic_ids"],
            batch["semantic_ids_len"],
            batch["bert_feature"],
        )
        self.manual_backward(loss)
        self._grad_accum_count += 1
        if self._grad_accum_count >= self.grad_accum_steps:
            self._step_accumulated_gradients(opt, scheduler)

        self.log(
            "total_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "lr",
            scheduler.get_last_lr()[0],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"top_{self.top_k}_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def _step_accumulated_gradients(self, opt, scheduler):
        opt.step()
        opt.zero_grad()
        scheduler.step()
        self._grad_accum_count = 0

    def on_train_epoch_end(self):
        # Flush a partial accumulation window instead of silently dropping the
        # final 1-3 batches when the epoch length is not divisible by the
        # configured accumulation count.
        if getattr(self, "_grad_accum_count", 0) <= 0:
            return
        self._step_accumulated_gradients(
            self.optimizers(),
            self.lr_schedulers(),
        )

    def validation_step(self, batch: Dict, batch_idx: int):
        return

    # # get loss
    # loss, acc = self.model.forward(
    #     batch['phoneme_ids'], batch['phoneme_ids_len'],
    #     batch['semantic_ids'], batch['semantic_ids_len'],
    #     batch['bert_feature']
    # )
    #
    # self.log(
    #     "val_total_loss",
    #     loss,
    #     on_step=True,
    #     on_epoch=True,
    #     prog_bar=True,
    #     sync_dist=True)
    # self.log(
    #     f"val_top_{self.top_k}_acc",
    #     acc,
    #     on_step=True,
    #     on_epoch=True,
    #     prog_bar=True,
    #     sync_dist=True)
    #
    # # get infer output
    # semantic_len = batch['semantic_ids'].size(1)
    # prompt_len = min(int(semantic_len * 0.5), 150)
    # prompt = batch['semantic_ids'][:, :prompt_len]
    # pred_semantic = self.model.infer(batch['phoneme_ids'],
    #                                  batch['phoneme_ids_len'], prompt,
    #                                  batch['bert_feature']
    #                                  )
    # save_name = f'semantic_toks_{batch_idx}.pt'
    # save_path = os.path.join(self.eval_dir, save_name)
    # torch.save(pred_semantic.detach().cpu(), save_path)

    def configure_optimizers(self):
        named_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if not named_parameters:
            raise RuntimeError(
                "S1 optimizer has no trainable parameters"
            )
        model_parameters = [
            parameter
            for _, parameter in named_parameters
        ]
        parameters_names = [
            [name for name, _ in named_parameters]
        ]
        lm_opt = ScaledAdam(
            model_parameters,
            lr=0.01,
            betas=(0.9, 0.95),
            clipping_scale=2.0,
            parameters_names=parameters_names,
            show_dominant_parameters=False,
            clipping_update_period=1000,
        )

        return {
            "optimizer": lm_opt,
            "lr_scheduler": {
                "scheduler": WarmupCosineLRSchedule(
                    lm_opt,
                    init_lr=self.config["optimizer"]["lr_init"],
                    peak_lr=self.config["optimizer"]["lr"],
                    end_lr=self.config["optimizer"]["lr_end"],
                    warmup_steps=self.config["optimizer"]["warmup_steps"],
                    total_steps=self.config["optimizer"]["decay_steps"],
                )
            },
        }
