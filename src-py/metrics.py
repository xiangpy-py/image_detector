import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from config import CLASS_NAMES
from dataset import get_class_counts


def get_pos_weight(labels, device):
    counts = get_class_counts(labels)
    normal = counts["NORMAL"]
    pneumonia = counts["PNEUMONIA"]
    pos_weight = torch.tensor([normal / pneumonia], dtype=torch.float32, device=device)
    return pos_weight


def evaluate_model(model, dataloader, device, criterion=None):
    """
    评估模型在给定数据集上的性能。

    Args:
        model: PyTorch 模型
        dataloader: DataLoader
        device: 计算设备 (cuda/cpu)
        criterion: 可选的损失函数；如果为 None，则不计算 loss

    Returns:
        metrics: 字典，包含各项评估指标
        all_labels: 真实标签数组
        all_probs: 预测概率数组
    """
    model.eval()
    all_labels = []
    all_probs = []
    total_loss = 0.0
    has_loss = criterion is not None
    n_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

            outputs = model(images)
            if has_loss:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            n_samples += images.size(0)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
    }
    if has_loss:
        metrics["loss"] = total_loss / n_samples

    return metrics, all_labels, all_probs


def get_criterion(device, pos_weight):
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
