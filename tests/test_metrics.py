"""测试 metrics 模块。"""

import numpy as np
import pytest
import torch

from metrics import evaluate_model, get_criterion, get_pos_weight


class FakeModel:
    """模拟二分类模型，用于测试 evaluate_model。"""

    def __init__(self, logits):
        self.logits = logits
        self._eval = False
        self._idx = 0

    def eval(self):
        self._eval = True
        self._idx = 0
        return self

    def __call__(self, images):
        bs = images.size(0)
        out = self.logits[self._idx : self._idx + bs]
        self._idx += bs
        return torch.tensor(out, dtype=torch.float32).unsqueeze(1)


def test_get_pos_weight():
    labels = np.array([0, 0, 1, 1, 1, 1])  # 2 NORMAL, 4 PNEUMONIA
    device = torch.device("cpu")
    pw = get_pos_weight(labels, device)
    assert pw.shape == (1,)
    assert pytest.approx(pw.item(), 0.001) == 2.0 / 4.0


def test_get_criterion():
    device = torch.device("cpu")
    pw = torch.tensor([0.5])
    crit = get_criterion(device, pw)
    assert crit is not None


def test_evaluate_model_basic():
    """测试 evaluate_model 基本功能。"""
    # 构造 4 个样本：2 正 2 负，模型完美预测
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 0, 1, 1])
    dataset = torch.utils.data.TensorDataset(images, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    # 构造 logits 使得 sigmoid 后 [0.1, 0.1, 0.9, 0.9]
    logits = np.array([-2.2, -2.2, 2.2, 2.2])
    model = FakeModel(logits)

    metrics, all_labels, all_probs = evaluate_model(model, loader, torch.device("cpu"))

    assert pytest.approx(metrics["accuracy"], 0.01) == 1.0
    assert pytest.approx(metrics["precision"], 0.01) == 1.0
    assert pytest.approx(metrics["recall"], 0.01) == 1.0
    assert pytest.approx(metrics["f1"], 0.01) == 1.0
    assert pytest.approx(metrics["auc"], 0.01) == 1.0
    assert len(all_labels) == 4
    assert len(all_probs) == 4


def test_evaluate_model_with_loss():
    """测试 evaluate_model 传入 criterion 时计算 loss。"""
    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])
    dataset = torch.utils.data.TensorDataset(images, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    logits = np.array([0.0, 0.0])  # sigmoid=0.5，半对半错
    model = FakeModel(logits)
    criterion = torch.nn.BCEWithLogitsLoss()

    metrics, _, _ = evaluate_model(model, loader, torch.device("cpu"), criterion)
    assert "loss" in metrics
    assert metrics["loss"] > 0


def test_evaluate_model_single_class():
    """测试所有标签相同的情况下 AUC 为 0。"""
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 0, 0, 0])
    dataset = torch.utils.data.TensorDataset(images, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    logits = np.array([0.1, 0.2, 0.3, 0.4])
    model = FakeModel(logits)

    metrics, _, _ = evaluate_model(model, loader, torch.device("cpu"))
    assert metrics["auc"] == 0.0  # 单类无法计算 AUC
