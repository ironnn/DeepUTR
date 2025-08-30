import torch
from sklearn.metrics import r2_score
import pytorch_lightning as pl
import torch.nn as nn
import os
import csv
from datetime import datetime
import torch.nn.functional as F
from colonmodel import jambaregression
import yaml

class colonmodule(pl.LightningModule):
    def __init__(self, d_model, num_mamba_blocks, d_intermediate, vocab_size, max_seq_len, lr):
        super().__init__()
        self.model = jambaregression(num_mamba_blocks, d_model, d_intermediate, vocab_size, max_seq_len)
        self.criterion = nn.MSELoss()
        self.lr = lr

        self.best_val_r2 = float("-inf")
        self.best_epoch = -1

        # 初始化属性，防止 AttributeError
        self.current_train_loss = None
        self.current_train_r2 = None

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.save_dir = f"results/results_{current_time}"
        os.makedirs(self.save_dir, exist_ok=True)

        self.metrics_csv = os.path.join(self.save_dir, "metrics.csv")
        self.best_outputs_csv = os.path.join(self.save_dir, "best_outputs.csv")

        if not os.path.exists(self.metrics_csv):
            with open(self.metrics_csv, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "train_loss", "train_r2", "val_loss", "val_r2"])

        if not os.path.exists(self.best_outputs_csv):
            with open(self.best_outputs_csv, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "rpf", "output"])

        self.train_outputs = []
        self.val_outputs = []

    def forward(self, batch):
        return self.model(
            batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
            tpm=batch.get("rna", None),
            cell_type=batch.get("cell_line_index", None)
        )

    def compute_r2(self, y_true, y_pred):
        y_true = y_true.detach().cpu().to(torch.float32).numpy()
        y_pred = y_pred.detach().cpu().to(torch.float32).numpy()
        return r2_score(y_true, y_pred)

    def weighted_loss(self, pred, rpf, soft_hard):
        loss = F.mse_loss(pred, rpf, reduction="none")
        return (loss * soft_hard).mean()

    def general_step(self, batch, batch_idx=None):
        y_true = batch["rpf"].float().unsqueeze(-1)   # 目标是 rpf
        pred = self(batch)
        loss = self.criterion(pred, y_true)
        r2 = self.compute_r2(y_true, pred)
        return loss, r2, y_true, pred

    def training_step(self, batch, batch_idx):
        loss, r2, rpf, output = self.general_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch["input_ids"].shape[0])
        self.log("train_r2_step", r2, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch["input_ids"].shape[0])
        self.train_outputs.append({"loss": loss, "r2": r2, "rpf": rpf, "output": output})
        return {"loss": loss, "r2": r2, "rpf": rpf, "output": output}

    def validation_step(self, batch, batch_idx):
        loss, r2, rpf, output = self.general_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=False, on_epoch=True, sync_dist=True, batch_size=batch["input_ids"].shape[0])
        self.log("val_r2", r2, prog_bar=False, on_epoch=True, sync_dist=True, batch_size=batch["input_ids"].shape[0])
        self.val_outputs.append({"loss": loss, "r2": r2, "rpf": rpf, "output": output})
        return {"loss": loss, "r2": r2, "rpf": rpf, "output": output}

    def on_train_epoch_end(self):
        if not self.train_outputs:
            return
        avg_loss = torch.stack([x["loss"] for x in self.train_outputs]).mean().item()
        avg_r2 = torch.tensor([x["r2"] for x in self.train_outputs]).mean().item()
        print(f"Epoch {self.current_epoch}: Train Loss: {avg_loss:.4f}, Train R²: {avg_r2:.4f}")
        self.current_train_loss = avg_loss
        self.current_train_r2 = avg_r2
        self.train_outputs.clear()

         # === 1. 保存权重 ===
        weights_dir = "./pure_weights_colon"
        os.makedirs(weights_dir, exist_ok=True)
        weight_path = os.path.join(weights_dir, f"epoch{self.current_epoch:02d}.pt")
        torch.save(self.model.state_dict(), weight_path)

        # === 2. 保存 config（自己整理想保留的参数）===
        config = {
            "d_model": self.model.d_model,
            "num_mamba_blocks": self.model.num_mamba_blocks,
            "d_intermediate": self.model.d_intermediate,
            "vocab_size": self.model.vocab_size,
            "max_seq_len": self.model.max_seq_len,
            "lr": self.lr,
            # 其它你想保存的参数
        }
        config_path = os.path.join(weights_dir, f"epoch{self.current_epoch:02d}_config.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return
        avg_loss = torch.stack([x["loss"] for x in self.val_outputs]).mean().item()
        avg_r2 = torch.tensor([x["r2"] for x in self.val_outputs]).mean().item()
        print(f"Epoch {self.current_epoch}: Val Loss: {avg_loss:.4f}, Val R²: {avg_r2:.4f}")

        if self.trainer.is_global_zero:
            if avg_r2 > self.best_val_r2:
                self.best_val_r2 = avg_r2
                self.best_epoch = self.current_epoch
                with open(self.best_outputs_csv, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["epoch", "rpf", "output"])
                    for output in self.val_outputs:
                        rpf = output["rpf"].detach().cpu().numpy().flatten()
                        pred = output["output"].detach().cpu().numpy().flatten()
                        for l, p in zip(rpf, pred):
                            writer.writerow([self.current_epoch, l, p])

            with open(self.metrics_csv, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    self.current_epoch,
                    self.current_train_loss if self.current_train_loss is not None else -1,
                    self.current_train_r2 if self.current_train_r2 is not None else -1,
                    avg_loss,
                    avg_r2
                ])

        self.val_outputs.clear()

    def configure_optimizers(self):
        param_groups = [
            {"params": [p for n, p in self.named_parameters() if "bias" not in n], "weight_decay": 1e-2},
            {"params": [p for n, p in self.named_parameters() if "bias" in n], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(param_groups, lr=self.lr)

    def setup(self, stage=None):
        if isinstance(self.trainer.strategy, pl.strategies.DDPStrategy):
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
