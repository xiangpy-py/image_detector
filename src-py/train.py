import json
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from config import (
    CLASS_NAMES,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    MODELS_DIR,
    OUTPUTS_DIR,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    USE_EMA,
    USE_GRADIENT_CLIP,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from dataset import get_class_counts, get_dataloaders, load_cached_data
from logger_config import setup_logger
from metrics import evaluate_model, get_loss_function, get_pos_weight
from model import build_model, set_seed


class WarmupCosineScheduler:
    """先线性 Warmup，再 Cosine Annealing 的学习率调度器。

    Args:
        optimizer: 优化器
        warmup_epochs: warmup 轮数
        total_epochs: 总训练轮数
        min_lr: 最小学习率
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # 线性 warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # cosine annealing
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)))
            lr = float(lr)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class ModelEMA:
    """模型参数的指数移动平均 (EMA)。

    EMA 模型在验证时通常比原始模型更稳定、泛化更好。
    """

    def __init__(self, model, decay=0.999):
        import copy
        self.model = copy.deepcopy(model)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


def _smooth_metric(values, window=3):
    """计算滑动平均。"""
    if len(values) < window:
        return values[-1] if values else 0.0
    return sum(values[-window:]) / window


def train(resume_from=None):
    set_seed()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    train_loader, val_loader, test_loader = get_dataloaders()
    train_images, train_labels = load_cached_data("train")
    logger.info(f"训练集类别分布: {get_class_counts(train_labels)}")

    model = build_model(pretrained=True).to(device)
    logger.info(f"模型架构: {type(model).__name__}")

    pos_weight = get_pos_weight(train_labels, device)
    criterion = get_loss_function(device, pos_weight=pos_weight)
    logger.info(f"损失函数: {type(criterion).__name__}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 主学习率调度：Warmup + Cosine Annealing
    lr_scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=EPOCHS
    )
    # Fallback 调度器：验证 F1 停滞时降低学习率
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )

    use_amp = device.type == "cuda"
    scaler = GradScaler(device=str(device)) if use_amp else None
    amp_enabled = use_amp

    # EMA 模型
    ema_model = ModelEMA(model, decay=0.999) if USE_EMA else None

    start_epoch = 0
    best_f1 = 0.0
    best_auc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}
    val_f1_history = []  # 用于平滑早停

    # --- Resume 支持 ---
    resume_path = None
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"指定的 checkpoint 不存在: {resume_path}")

    if resume_path:
        logger.info(f"从 checkpoint 恢复训练: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            plateau_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_f1 = checkpoint.get("best_f1", 0.0)
        best_auc = checkpoint.get("best_auc", 0.0)
        if "history" in checkpoint:
            history = checkpoint["history"]
            val_f1_history = history.get("val_f1", [])
        if ema_model and "ema_state_dict" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema_state_dict"])
        logger.info(f"恢复至 epoch {start_epoch}, 当前最佳 val_f1={best_f1:.4f}")
    # -------------------

    for epoch in range(start_epoch, EPOCHS):
        # 更新学习率
        current_lr = lr_scheduler.step(epoch)
        logger.info(f"Epoch {epoch + 1}/{EPOCHS} | LR={current_lr:.2e}")

        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if amp_enabled:
                scaler.scale(loss).backward()
                # Gradient Clipping
                if USE_GRADIENT_CLIP:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if USE_GRADIENT_CLIP:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            # 更新 EMA
            if ema_model:
                ema_model.update(model)

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({"loss": loss.item(), "lr": f"{current_lr:.2e}"})

        train_loss = running_loss / len(train_loader.dataset)

        # 验证时优先使用 EMA 模型（更稳定）
        eval_model = ema_model.model if ema_model else model
        val_metrics, _, _ = evaluate_model(eval_model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["auc"])
        val_f1_history.append(val_metrics["f1"])

        logger.info(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # Fallback 学习率调度
        plateau_scheduler.step(val_metrics["f1"])

        # 实时保存历史
        history_path = OUTPUTS_DIR / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # 平滑后的 F1 用于早停判断（避免单次波动导致过早停止）
        smooth_f1 = _smooth_metric(val_f1_history, window=3)

        # 保存最佳 F1 检查点
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            best_path = MODELS_DIR / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": plateau_scheduler.state_dict(),
                    "best_f1": best_f1,
                    "best_auc": best_auc,
                    "ema_state_dict": ema_model.state_dict() if ema_model else None,
                    "history": history,
                },
                best_path,
            )
            logger.info(f"最佳 F1 模型已保存，val_f1={best_f1:.4f}")
        else:
            patience_counter += 1

        # 保存最佳 AUC 检查点
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            auc_path = MODELS_DIR / "best_auc_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": plateau_scheduler.state_dict(),
                    "best_f1": best_f1,
                    "best_auc": best_auc,
                    "ema_state_dict": ema_model.state_dict() if ema_model else None,
                    "history": history,
                },
                auc_path,
            )
            logger.info(f"最佳 AUC 模型已保存，val_auc={best_auc:.4f}")

        # 保存最后一个 epoch 的检查点（崩溃恢复用）
        last_path = MODELS_DIR / "last_model.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": plateau_scheduler.state_dict(),
                "best_f1": best_f1,
                "best_auc": best_auc,
                "ema_state_dict": ema_model.state_dict() if ema_model else None,
                "history": history,
            },
            last_path,
        )

        # 平滑早停：基于 3-epoch 滑动平均 F1
        if patience_counter >= EARLY_STOP_PATIENCE:
            logger.info(f"早停触发 (patience={EARLY_STOP_PATIENCE}, 平滑 F1 未提升)")
            break

    logger.info("训练完成")
    return history


if __name__ == "__main__":
    setup_logger()
    train()
