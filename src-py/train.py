import json
from collections import deque
from datetime import datetime
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
    FREEZE_BACKBONE,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    MODELS_DIR,
    MODEL_ARCH,
    OUTPUTS_DIR,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    UNFREEZE_EPOCH,
    USE_EMA,
    USE_GRADIENT_CLIP,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from dataset import get_class_counts, get_dataloaders, load_cached_data
from logger_config import setup_logger
from metrics import evaluate_model, get_loss_function, get_pos_weight
from model import build_model, set_seed, unfreeze_model


class WarmupCosineScheduler:
    """先线性 Warmup，再 Cosine Annealing 的学习率调度器。"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)))
            lr = float(lr)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class ModelEMA:
    """模型参数的指数移动平均 (EMA)。"""

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


def _count_trainable_params(model):
    """统计可训练参数数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(resume_from=None, overrides: dict | None = None):
    """运行训练流程。

    Args:
        resume_from: 恢复训练的 checkpoint 路径
        overrides: 覆盖 config.py 中的超参（仅本次训练生效），
                   支持 keys: epochs, lr, weight_decay, patience
    """
    overrides = overrides or {}
    epochs = int(overrides.get("epochs", EPOCHS))
    lr = float(overrides.get("lr", LEARNING_RATE))
    weight_decay = float(overrides.get("weight_decay", WEIGHT_DECAY))
    patience = int(overrides.get("patience", EARLY_STOP_PATIENCE))

    set_seed()

    # 生成本次训练的唯一时间戳前缀，用于区分不同训练运行的模型
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    logger.info(f"训练时间戳: {run_timestamp}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    train_loader, val_loader, test_loader = get_dataloaders()
    train_images, train_labels = load_cached_data("train")
    logger.info(f"训练集类别分布: {get_class_counts(train_labels)}")

    # ─── 两阶段训练：阶段 1 冻结 backbone ───
    model = build_model(pretrained=True, freeze_backbone=FREEZE_BACKBONE).to(device)
    logger.info(f"模型架构: {type(model).__name__}")
    logger.info(f"可训练参数: {_count_trainable_params(model):,} / {sum(p.numel() for p in model.parameters()):,}")
    if FREEZE_BACKBONE:
        logger.info(f"阶段 1: 冻结 backbone，将在 epoch {UNFREEZE_EPOCH} 后解冻")

    pos_weight = get_pos_weight(train_labels, device)
    criterion = get_loss_function(device, pos_weight=pos_weight)
    logger.info(f"损失函数: {type(criterion).__name__}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    lr_scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=epochs
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )

    use_amp = device.type == "cuda"
    scaler = GradScaler(device=str(device)) if use_amp else None
    amp_enabled = use_amp

    ema_model = ModelEMA(model, decay=0.999) if USE_EMA else None

    start_epoch = 0
    best_f1 = 0.0
    best_auc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}
    val_f1_history = []

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

    for epoch in range(start_epoch, epochs):
        # ─── 两阶段：在 UNFREEZE_EPOCH 时解冻 backbone ───
        if FREEZE_BACKBONE and epoch == UNFREEZE_EPOCH:
            unfreeze_model(model)
            logger.info(f"🔄 Epoch {epoch + 1}: 解冻 backbone，开始微调全部参数")
            logger.info(f"可训练参数: {_count_trainable_params(model):,}")

            # 重建 optimizer：避免 AdamW momentum 污染，并降低微调学习率
            new_lr = lr / 10
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=new_lr, weight_decay=weight_decay
            )
            # 新的 lr_scheduler：微调阶段从 new_lr 开始 cosine 下降，无 warmup
            lr_scheduler = WarmupCosineScheduler(
                optimizer, warmup_epochs=0, total_epochs=epochs - UNFREEZE_EPOCH, min_lr=1e-7
            )
            plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
            )
            # 重置 EMA 避免旧参数统计干扰
            if ema_model:
                ema_model = ModelEMA(model, decay=0.999)
            logger.info(f"Optimizer 已重建，微调学习率: {new_lr:.2e}")

        # 解冻后使用相对 epoch 计算学习率
        if FREEZE_BACKBONE and epoch >= UNFREEZE_EPOCH:
            current_lr = lr_scheduler.step(epoch - UNFREEZE_EPOCH)
        else:
            current_lr = lr_scheduler.step(epoch)
        phase = "微调" if (not FREEZE_BACKBONE or epoch >= UNFREEZE_EPOCH) else "冻结"
        logger.info(f"Epoch {epoch + 1}/{epochs} | LR={current_lr:.2e} | 阶段: {phase}")

        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if amp_enabled:
                scaler.scale(loss).backward()
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

            if ema_model:
                ema_model.update(model)

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({"loss": loss.item(), "lr": f"{current_lr:.2e}"})

        train_loss = running_loss / len(train_loader.dataset)

        eval_model = ema_model.model if ema_model else model
        val_metrics, _, _ = evaluate_model(eval_model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["auc"])
        val_f1_history.append(val_metrics["f1"])

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        plateau_scheduler.step(val_metrics["f1"])

        # 写入 history.json：临时文件 + 原子 rename，避免崩溃时损坏
        history_path = OUTPUTS_DIR / "history.json"
        tmp_path = history_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        tmp_path.replace(history_path)

        smooth_f1 = _smooth_metric(val_f1_history, window=3)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            best_path = MODELS_DIR / f"{run_timestamp}_best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "arch": MODEL_ARCH,
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

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            auc_path = MODELS_DIR / f"{run_timestamp}_best_auc_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "arch": MODEL_ARCH,
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

        last_path = MODELS_DIR / f"{run_timestamp}_last_model.pth"
        torch.save(
            {
                "epoch": epoch,
                "arch": MODEL_ARCH,
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

        if patience_counter >= patience:
            logger.info(f"早停触发 (patience={patience}, 平滑 F1 未提升)")
            break

    logger.info("训练完成")
    return history


if __name__ == "__main__":
    setup_logger()
    train()
