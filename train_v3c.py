#!/usr/bin/env python3
"""
VballNetV3c Training Script
Specialized training script for VballNetV3c model variants

Usage Examples:
python train_v3c.py --data dataset/train
python train_v3c.py --data dataset/train --batch 8 --epochs 50 --lr 0.001
python train_v3c.py --data dataset/train --optimizer Adam --lr 0.001 --batch 16 --plot 10
python train_v3c.py --resume best.pth --data dataset/train --lr 0.0001 --reset-lr
python train_v3c.py --resume checkpoint.pth --data dataset/train --optimizer Adam --epochs 100

Parameters:
--data: Training dataset path (required)
--resume: Checkpoint path for resuming
--reset-lr: If set, use --lr from command line even when resuming (default: False)
--split: Train/val split ratio (default: 0.8)
--seed: Random seed (default: 26)
--batch: Batch size (default: 3)
--epochs: Training epochs (default: 30)
--workers: Data loader workers (default: 0)
--device: Device auto/cpu/cuda/mps (default: auto)
--optimizer: Adadelta/Adam/AdamW/SGD (default: Adadelta)
--lr: Learning rate (default: auto per optimizer)
--wd: Weight decay (default: 0)
--scheduler: ReduceLROnPlateau/None (default: ReduceLROnPlateau)
--factor: LR reduction factor (default: 0.5)
--patience: LR reduction patience (default: 3)
--min_lr: Minimum learning rate (default: 1e-6)
--plot: Loss plot interval (default: 1)
--out: Output directory (default: outputs)
--name: Experiment name (default: exp)
--model_variant: Which VballNetV3c variant to train (default: original)
"""

import argparse
import json
import signal
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model.loss import WeightedBinaryCrossEntropy
from preprocessing.tracknet_datasetv2 import FrameHeatmapDataset
import os
import numpy as np
import cv2

# Import only VballNetV3c model variants
from model.vballnet_v3c import VballNetV3b as VballNetV3cOriginal
from model.vballnet_v3c_minimal import VballNetV3cMinimal
from model.vballnet_v3c_improved import VballNetV3cImproved
from model.vballnet_v3c_enhanced import VballNetV3cEnhanced
from model.vballnet_v3c_optimized import VballNetV3cOptimized


def parse_args():
    parser = argparse.ArgumentParser(description="VballNetV3c Training")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--reset-lr", action="store_true", help="Use --lr from command line even when resuming")
    parser.add_argument("--val_data", type=str, default=None, help="Validation dataset path (optional)")
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adadelta",
        choices=["Adadelta", "Adam", "AdamW", "SGD"],
    )
    parser.add_argument("--lr", type=float)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ReduceLROnPlateau",
        choices=["ReduceLROnPlateau", "None"],
    )
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--plot", type=int, default=1)
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument(
        "--model_variant",
        type=str,
        default="original",
        choices=["original", "minimal", "improved", "enhanced", "optimized"],
        help="Which VballNetV3c variant to train"
    )
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--seq', type=int, default=9)
    parser.add_argument('--alpha', type=float, default=-1, help='alpha of sample mixup, -1 means no mixup')

    args = parser.parse_args()

    if args.lr is None:
        lr_defaults = {"Adadelta": 1.0, "Adam": 0.001, "AdamW": 0.001, "SGD": 0.01}
        args.lr = lr_defaults[args.optimizer]

    return args


def mixup(x, y, alpha=0.5):
    batch_size = x.size()[0]
    lamb = np.random.beta(alpha, alpha, size=batch_size)
    lamb = np.maximum(lamb, 1 - lamb)
    lamb = torch.from_numpy(lamb[:, None, None, None]).float().to(x.device)
    index = torch.randperm(batch_size)
    x_mix = x * lamb + x[index] * (1 - lamb)
    y_mix = y * lamb + y[index] * (1 - lamb)
    return x_mix, y_mix


class VballNetV3cTrainer:
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.interrupted = False
        self.best_loss = float("inf")
        self.device = self._get_device()
        self._setup_dirs()
        self._load_checkpoint()
        self.losses = {"batch": [], "steps": [], "lrs": [], "train": [], "val": []}
        self.step = 0
        signal.signal(signal.SIGINT, self._interrupt)
        signal.signal(signal.SIGTERM, self._interrupt)

    def _get_device(self):
        if self.args.device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(self.args.device)

    def _setup_dirs(self):
        print("Setting up output directories...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_resumed" if self.args.resume else ""
        model_tag = f"VballNetV3c_{self.args.model_variant}_seq{self.args.seq}" + ("_grayscale" if self.args.grayscale else "")
        self.save_dir = Path(self.args.out) / f"{self.args.name}_{model_tag}{suffix}_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "checkpoints").mkdir(exist_ok=True)
        (self.save_dir / "plots").mkdir(exist_ok=True)
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(vars(self.args), f, indent=2)
        print(f"Output directory created: {self.save_dir}")

    def _load_checkpoint(self):
        if not self.args.resume:
            return
        print("Loading checkpoint...")
        path = Path(self.args.resume)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        self.checkpoint = torch.load(path, map_location="cpu")
        self.start_epoch = self.checkpoint["epoch"] + (
            0 if self.checkpoint.get("is_emergency", False) else 1
        )
        self.best_loss = self.checkpoint.get("best_loss", float("inf"))
        self.step = self.checkpoint.get("step", 0)
        self.losses = self.checkpoint.get("history", {"batch": [], "steps": [], "lrs": [], "train": [], "val": []})
        print(f"Checkpoint loaded, resuming from epoch \033[93m{self.start_epoch + 1}\033[0m")

    def _interrupt(self, signum, frame):
        print("\n\033[91mInterrupt detected\033[0m, saving emergency checkpoint...")
        self.interrupted = True

    def _calculate_effective_lr(self):
        if self.args.optimizer == "Adadelta":
            if not hasattr(self.optimizer, "state") or not self.optimizer.state:
                return self.args.lr
            effective_lrs = []
            eps = self.optimizer.param_groups[0].get("eps", 1e-6)
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.optimizer.state[p]
                    if len(state) == 0:
                        continue
                    square_avg = state.get("square_avg")
                    acc_delta = state.get("acc_delta")
                    if square_avg is not None and acc_delta is not None:
                        if torch.is_tensor(square_avg) and torch.is_tensor(acc_delta):
                            rms_delta = (acc_delta + eps).sqrt().mean()
                            rms_grad = (square_avg + eps).sqrt().mean()
                            if rms_grad > eps:
                                effective_lr = self.args.lr * rms_delta / rms_grad
                                effective_lrs.append(effective_lr.item())
            if effective_lrs:
                avg_lr = sum(effective_lrs) / len(effective_lrs)
                return max(avg_lr, eps)
            else:
                return self.args.lr
        else:
            return self.optimizer.param_groups[0]["lr"]

    def setup_data(self):
        print("Loading dataset...")
        dataset = FrameHeatmapDataset(
            self.args.data,
            seq=self.args.seq,
            grayscale=self.args.grayscale
        )
        print(f"Dataset loaded: \033[94m{len(dataset)}\033[0m samples {self.args.grayscale} sequences  {self.args.seq}")

        if self.args.val_data:
            print("Loading separate validation dataset...")
            val_dataset = FrameHeatmapDataset(
                self.args.val_data,
                seq=self.args.seq,
                grayscale=self.args.grayscale
            )
            train_ds = dataset
            val_ds = val_dataset
        else:
            print("Splitting dataset...")
            torch.manual_seed(self.args.seed)
            train_size = int(self.args.split * len(dataset))
            train_ds, val_ds = random_split(
                dataset, [train_size, len(dataset) - train_size]
            )

        print("Creating data loaders...")
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=self.device.type == "cuda",
        )

        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.args.batch,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=self.device.type == "cuda",
        )
        print(
            f"Data loaders ready - Train: \033[94m{len(train_ds)}\033[0m | Val: \033[94m{len(val_ds)}\033[0m"
        )

    def _create_optimizer(self):
        optimizers = {
            "Adadelta": lambda: torch.optim.Adadelta(
                self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd
            ),
            "Adam": lambda: torch.optim.Adam(
                self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd
            ),
            "AdamW": lambda: torch.optim.AdamW(
                self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd
            ),
            "SGD": lambda: torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.wd,
                momentum=0.9,
            ),
        }
        return optimizers[self.args.optimizer]()

    def setup_model(self):
        print("Initializing model...")
        if self.args.grayscale:
            in_dim = self.args.seq
            out_dim = self.args.seq
        else:
            in_dim = self.args.seq * 3
            out_dim = self.args.seq

        # Select the appropriate model variant
        if self.args.model_variant == "original":
            self.model = VballNetV3cOriginal(
                height=288,
                width=512,
                in_dim=in_dim,
                out_dim=out_dim,
            ).to(self.device)
            model_name = "VballNetV3cOriginal"
        elif self.args.model_variant == "minimal":
            self.model = VballNetV3cMinimal(
                height=288,
                width=512,
                in_dim=in_dim,
                out_dim=out_dim,
            ).to(self.device)
            model_name = "VballNetV3cMinimal"
        elif self.args.model_variant == "improved":
            self.model = VballNetV3cImproved(
                height=288,
                width=512,
                in_dim=in_dim,
                out_dim=out_dim,
            ).to(self.device)
            model_name = "VballNetV3cImproved"
        elif self.args.model_variant == "enhanced":
            self.model = VballNetV3cEnhanced(
                height=288,
                width=512,
                in_dim=in_dim,
                out_dim=out_dim,
            ).to(self.device)
            model_name = "VballNetV3cEnhanced"
        elif self.args.model_variant == "optimized":
            self.model = VballNetV3cOptimized(
                height=288,
                width=512,
                in_dim=in_dim,
                out_dim=out_dim,
            ).to(self.device)
            model_name = "VballNetV3cOptimized"
        else:
            raise ValueError(f"Unknown model variant: {self.args.model_variant}")

        self.criterion = WeightedBinaryCrossEntropy()
        self.optimizer = self._create_optimizer()

        if self.args.scheduler == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.args.factor,
                patience=self.args.patience,
                min_lr=self.args.min_lr,
            )
        else:
            self.scheduler = None

        # Load state from checkpoint
        if hasattr(self, "checkpoint"):
            print("Loading model, optimizer, and scheduler states from checkpoint...")
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
            
            if self.scheduler is not None and "scheduler_state_dict" in self.checkpoint:
                self.scheduler.load_state_dict(self.checkpoint["scheduler_state_dict"])
            
            # Force LR update if --reset-lr is specified
            if self.args.reset_lr and self.args.lr is not None:
                print(f"\033[93m--reset-lr enabled: overriding learning rate to {self.args.lr}\033[0m")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.args.lr
            else:
                # Restore args.lr from optimizer state for correct display
                self.args.lr = self.optimizer.param_groups[0]["lr"]

            print("All states loaded successfully")

        print(
            f"Model ready - Variant: \033[93m{model_name}\033[0m | Optimizer: \033[93m{self.args.optimizer}\033[0m | LR: \033[93m{self.args.lr}\033[0m | WD: \033[93m{self.args.wd}\033[0m"
        )

    def save_checkpoint(self, epoch, train_loss, val_loss, is_emergency=False):
        print("Saving checkpoint...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_tag = f"VballNetV3c_{self.args.model_variant}_seq{self.args.seq}" + ("_grayscale" if self.args.grayscale else "")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "is_emergency": is_emergency,
            "history": self.losses.copy(),
            "step": self.step,
            "timestamp": timestamp,
            "best_loss": self.best_loss,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        prefix = "emergency_" if is_emergency else "checkpoint_"
        filename = f"{prefix}{model_tag}_epoch_{epoch + 1}_{timestamp}.pth"
        filepath = self.save_dir / "checkpoints" / filename
        torch.save(checkpoint, filepath)

        if not is_emergency and val_loss < self.best_loss:
            self.best_loss = val_loss
            best_name = f"{model_tag}_best.pth"
            torch.save(checkpoint, self.save_dir / "checkpoints" / best_name)
            print(f"Checkpoint saved: {filename} (\033[92mBest model updated\033[0m)")
            return filepath, True

        print(f"Checkpoint saved: {filename}")
        return filepath, False

    def plot_curves(self, epoch):
        print("Generating training plots...")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

        if self.losses["train"]:
            cnt = len(self.losses["train"])
            epochs = list(range(1, cnt + 1))
            ax1.plot(epochs, self.losses["train"], "bo-", label="Train Loss")
            ax1.plot(epochs, self.losses["val"], "ro-", label="Val Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Train and Validation Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        if self.losses["batch"]:
            ax2.plot(self.losses["steps"], self.losses["batch"], "b-", alpha=0.3, label="Batch Loss")
            ax2.set_xlabel("Batch")
            ax2.set_ylabel("Loss")
            ax2.set_title("Batch Loss")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        if self.losses["lrs"]:
            ax3.plot(self.losses["steps"], self.losses["lrs"], "g-")
            ax3.set_xlabel("Batch")
            ax3.set_ylabel("Learning Rate")
            ax3.set_title("Learning Rate")
            ax3.set_yscale("log")
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.save_dir / "plots" / f"training_metrics_epoch_{epoch + 1}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Training plots saved for epoch \033[93m{epoch + 1}\033[0m")

    def validate(self):
        print("Starting validation...")
        self.model.eval()
        total_loss = 0.0
        vis_dir = self.save_dir / "val_vis"
        vis_dir.mkdir(exist_ok=True)
        max_vis_batches = 5

        with torch.no_grad():
            val_pbar = tqdm(total=len(self.val_loader), desc="Validation", ncols=100)
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                if self.interrupted:
                    val_pbar.close()
                    break
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                model_output = self.model(inputs)

                if isinstance(model_output, tuple):
                    outputs = model_output[0]
                else:
                    outputs = model_output

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                if batch_idx < max_vis_batches:
                    inp = inputs[0].detach().cpu()
                    pred = outputs[0].detach().cpu()
                    gt = targets[0].detach().cpu()
                    n_vis = min(pred.shape[0], gt.shape[0], 9)
                    for i in range(n_vis):
                        if inp.shape[0] == pred.shape[0]:
                            rgb = np.stack([inp[i].numpy()] * 3, axis=2)
                        else:
                            rgb = inp[i*3:(i+1)*3].permute(1, 2, 0).numpy()
                        rgb = (rgb * 255).astype(np.uint8)
                        pred_hm = pred[i].numpy()
                        pred_hm = (pred_hm * 255).astype(np.uint8)
                        pred_hm_color = cv2.applyColorMap(pred_hm, cv2.COLORMAP_JET)
                        gt_hm = gt[i].numpy()
                        gt_hm = (gt_hm * 255).astype(np.uint8)
                        gt_hm_color = cv2.applyColorMap(gt_hm, cv2.COLORMAP_JET)
                        overlay_pred = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.6, pred_hm_color, 0.4, 0)
                        overlay_gt = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.6, gt_hm_color, 0.4, 0)
                        vis_img = np.vstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), overlay_pred, overlay_gt])
                        vis_path = vis_dir / f"val_batch{batch_idx}_frame{i}.jpg"
                        cv2.imwrite(str(vis_path), vis_img)

                val_pbar.update(1)
                val_pbar.set_postfix({"loss": f"{loss.item():.6f}"})
            val_pbar.close()

        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation completed - Average loss: \033[94m{avg_loss:.6f}\033[0m")
        return avg_loss

    def train(self):
        print(f"Starting training on \033[93m{self.device}\033[0m")
        self.setup_data()
        self.setup_model()

        for epoch in range(self.start_epoch, self.args.epochs):
            if self.interrupted:
                break

            print(f"\nEpoch \033[95m{epoch + 1}\033[0m/\033[95m{self.args.epochs}\033[0m")
            start_time = time.time()
            self.model.train()
            total_loss = 0.0
            train_pbar = tqdm(total=len(self.train_loader), desc=f"Training", ncols=100)

            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                if self.interrupted:
                    train_pbar.close()
                    print("Emergency save triggered...")
                    val_loss = self.validate()
                    self.save_checkpoint(epoch, total_loss / (batch_idx + 1), val_loss, True)
                    self.plot_curves(epoch)
                    return

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if self.args.alpha is not None and self.args.alpha > 0:
                    inputs, targets = mixup(inputs, targets, self.args.alpha)

                self.optimizer.zero_grad()

                model_output = self.model(inputs)

                if isinstance(model_output, tuple):
                    outputs, aux_loss = model_output
                else:
                    outputs = model_output
                    aux_loss = torch.tensor(0.0, device=outputs.device)

                main_loss = self.criterion(outputs, targets)
                loss = main_loss + aux_loss

                loss.backward()
                self.optimizer.step()

                batch_loss = loss.item()
                total_loss += batch_loss
                self.step += 1

                current_lr = self._calculate_effective_lr()

                if self.step % self.args.plot == 0:
                    self.losses["batch"].append(batch_loss)
                    self.losses["steps"].append(self.step)
                    self.losses["lrs"].append(current_lr)

                train_pbar.update(1)
                train_pbar.set_postfix({"loss": f"{batch_loss:.6f}", "lr": f"{current_lr:.2e}"})
            train_pbar.close()

            train_loss = total_loss / len(self.train_loader)
            val_loss = self.validate()

            self.losses["train"].append(train_loss)
            self.losses["val"].append(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time

            print(
                f"Epoch [\033[95m{epoch + 1}\033[0m/\033[95m{self.args.epochs}\033[0m] "
                f"Train: \033[94m{train_loss:.6f}\033[0m Val: \033[94m{val_loss:.6f}\033[0m "
                f"LR: \033[94m{current_lr:.6e}\033[0m Time: \033[94m{elapsed:.1f}s\033[0m"
            )

            if self.scheduler:
                print("Updating learning rate scheduler...")
                self.scheduler.step(val_loss)

            _, is_best = self.save_checkpoint(epoch, train_loss, val_loss)
            if is_best:
                print(f"\033[92mNew best model! Val Loss: {val_loss:.6f}\033[0m")

            self.plot_curves(epoch)

        if not self.interrupted:
            print("\n\033[92mTraining completed successfully!\033[0m")
            print(f"\033[92mAll results saved to: {self.save_dir}\033[0m")


if __name__ == "__main__":
    args = parse_args()
    trainer = VballNetV3cTrainer(args)
    trainer.train()