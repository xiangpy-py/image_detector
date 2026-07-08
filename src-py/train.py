import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import (
    CLASS_NAMES,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    MODELS_DIR,
    OUTPUTS_DIR,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    WEIGHT_DECAY,
)
from dataset import get_class_counts, get_dataloaders, load_cached_data
from model import build_model, set_seed


def get_pos_weight(labels, device):
    counts = get_class_counts(labels)
    normal = counts["NORMAL"]
    pneumonia = counts["PNEUMONIA"]
    pos_weight = torch.tensor([normal / pneumonia], dtype=torch.float32, device=device)
    return pos_weight


def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    all_labels = []
    all_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    metrics = {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
    }
    return metrics, all_labels, all_probs


def train():
    set_seed()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_loader, val_loader, test_loader = get_dataloaders()
    train_images, train_labels = load_cached_data("train")
    print(f"训练集类别分布: {get_class_counts(train_labels)}")

    model = build_model(pretrained=True).to(device)

    pos_weight = get_pos_weight(train_labels, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )

    use_amp = device.type == "cuda"
    scaler = GradScaler(device=str(device)) if use_amp else None
    amp_enabled = use_amp

    best_f1 = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}

    for epoch in range(EPOCHS):
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
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({"loss": loss.item()})

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics, _, _ = evaluate_model(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["auc"])

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        scheduler.step(val_metrics["f1"])

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            best_path = MODELS_DIR / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                },
                best_path,
            )
            print(f"最佳模型已保存，val_f1={best_f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print("早停触发")
            break

    history_path = OUTPUTS_DIR / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"训练历史已保存至 {history_path}")

    return history


if __name__ == "__main__":
    train()
