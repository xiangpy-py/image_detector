import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from config import CLASS_NAMES, MODELS_DIR, OUTPUTS_DIR
from dataset import get_dataloaders
from model import build_model
from train import evaluate_model


def get_criterion(device):
    return torch.nn.BCEWithLogitsLoss()


def evaluate_split(model, dataloader, device, split_name):
    criterion = get_criterion(device)
    metrics, labels, probs = evaluate_model(model, dataloader, device, criterion)

    preds = (np.array(probs) >= 0.5).astype(int)
    print(
        f"[{split_name}] "
        f"Accuracy={metrics['accuracy']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"AUC={metrics['auc']:.4f}"
    )
    return metrics, labels, probs


def plot_roc(labels, probs, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"ROC 曲线已保存至 {save_path}")


def plot_confusion_matrix(labels, probs, save_path):
    preds = (np.array(probs) >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"混淆矩阵已保存至 {save_path}")


def plot_training_history(history_path, save_path):
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["val_f1"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.title("Validation F1")

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["val_auc"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.title("Validation AUC")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"训练历史图已保存至 {save_path}")


def run_evaluation():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, test_loader = get_dataloaders()

    model_path = MODELS_DIR / "best_model.pth"
    model = build_model(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    val_metrics, val_labels, val_probs = evaluate_split(
        model, val_loader, device, "Val"
    )
    test_metrics, test_labels, test_probs = evaluate_split(
        model, test_loader, device, "Test"
    )

    plot_roc(test_labels, test_probs, OUTPUTS_DIR / "roc_curve.png")
    plot_confusion_matrix(
        test_labels, test_probs, OUTPUTS_DIR / "confusion_matrix.png"
    )

    history_path = OUTPUTS_DIR / "history.json"
    if history_path.exists():
        plot_training_history(history_path, OUTPUTS_DIR / "training_history.png")

    report = {
        "val": val_metrics,
        "test": test_metrics,
    }
    report_path = OUTPUTS_DIR / "metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"评估指标已保存至 {report_path}")


if __name__ == "__main__":
    run_evaluation()
