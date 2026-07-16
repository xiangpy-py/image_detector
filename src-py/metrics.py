import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from config import CLASS_NAMES, FOCAL_ALPHA, FOCAL_GAMMA, LABEL_SMOOTHING, LOSS_TYPE
from dataset import get_class_counts


class FocalLoss(nn.Module):
    """Focal Loss：降低易分类样本的权重，聚焦于难例。

    在类别不平衡场景中，Focal Loss 比加权 BCE 更能处理 hard negatives。

    Args:
        alpha: 正样本权重平衡因子 (0.25 表示正样本权重为 0.25)
        gamma: 聚焦参数 (γ=2 时易例损失大幅衰减)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs: raw logits, targets: float [0, 1]
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        # targets 为 1 时 alpha_t = alpha，为 0 时 alpha_t = 1 - alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        return loss.mean()


class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    """带 Label Smoothing 的 BCE Loss。

    将硬标签 0/1 替换为平滑后的 soft 标签，防止模型过度自信。
    """

    def __init__(self, smoothing=0.1, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        # 将硬标签 [0, 1] 平滑为 [smoothing/2, 1 - smoothing/2]
        smooth_targets = targets * (1 - self.smoothing) + self.smoothing / 2
        return F.binary_cross_entropy_with_logits(
            inputs, smooth_targets, pos_weight=self.pos_weight
        )


def get_pos_weight(labels, device):
    counts = get_class_counts(labels)
    normal = counts["NORMAL"]
    pneumonia = counts["PNEUMONIA"]
    pos_weight = torch.tensor([normal / pneumonia], dtype=torch.float32, device=device)
    return pos_weight


def get_loss_function(device, pos_weight=None, loss_type=None):
    """获取损失函数工厂。

    Args:
        device: torch device
        pos_weight: BCE 的正样本权重
        loss_type: 'bce' | 'focal'，None 则用 config.LOSS_TYPE

    Returns:
        nn.Module: 损失函数实例
    """
    from config import FOCAL_ALPHA, FOCAL_GAMMA, LOSS_TYPE

    loss_type = loss_type or LOSS_TYPE

    if loss_type == "focal":
        return FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    elif loss_type == "bce":
        return LabelSmoothingBCEWithLogitsLoss(
            smoothing=LABEL_SMOOTHING, pos_weight=pos_weight
        )
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")


def evaluate_model(model, dataloader, device, criterion=None, threshold=0.5):
    """在 dataloader 上评估二分类模型。

    Args:
        model: 评估模型（应为 eval 模式）
        dataloader: 数据加载器
        device: torch device
        criterion: 损失函数，None 则不计算 loss
        threshold: 将概率二值化的阈值，默认 0.5
            训练时使用 0.5 保证各 epoch 指标可比；
            最终评估时由调用方传入最优阈值。
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
    all_preds = (all_probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
        "threshold": float(threshold),
    }
    if has_loss:
        metrics["loss"] = total_loss / n_samples

    return metrics, all_labels, all_probs


def get_criterion(device, pos_weight):
    """兼容旧接口，返回与 get_loss_function(bce) 等价的损失函数。"""
    return get_loss_function(device, pos_weight=pos_weight, loss_type="bce")
