"""测试 dataset 模块。"""

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from dataset import CachedDataset, get_class_counts


def test_cached_dataset_len():
    images = np.random.randint(0, 256, (10, 3, 256, 256), dtype=np.uint8)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    ds = CachedDataset(images, labels)
    assert len(ds) == 10


def test_cached_dataset_getitem():
    images = np.random.randint(0, 256, (5, 3, 256, 256), dtype=np.uint8)
    labels = np.array([0, 0, 1, 1, 1])
    ds = CachedDataset(images, labels)

    img, lbl = ds[0]
    # 无 transform 时返回 PIL Image
    assert isinstance(img, Image.Image)
    assert img.size == (256, 256)
    assert lbl.item() == 0


def test_cached_dataset_with_transform():
    images = np.random.randint(0, 256, (3, 3, 256, 256), dtype=np.uint8)
    labels = np.array([0, 1, 0])

    ds = CachedDataset(images, labels, transform=transforms.ToTensor())
    img, _ = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 256, 256)


def test_cached_dataset_full_pipeline():
    """测试完整的 ToTensor + Normalize 流水线。"""
    images = np.random.randint(0, 256, (2, 3, 256, 256), dtype=np.uint8)
    labels = np.array([0, 1])

    pipeline = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = CachedDataset(images, labels, transform=pipeline)
    img, lbl = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    # 经过 Normalize 后值域应超出 [0, 1]
    assert img.min() < 0 or img.max() > 1


def test_get_class_counts():
    labels = np.array([0, 0, 0, 1, 1])
    counts = get_class_counts(labels)
    assert counts["NORMAL"] == 3
    assert counts["PNEUMONIA"] == 2


def test_get_class_counts_empty():
    labels = np.array([], dtype=int)
    counts = get_class_counts(labels)
    assert counts["NORMAL"] == 0
    assert counts["PNEUMONIA"] == 0


def test_get_class_counts_all_normal():
    labels = np.array([0, 0, 0])
    counts = get_class_counts(labels)
    assert counts["NORMAL"] == 3
    assert counts["PNEUMONIA"] == 0


def test_get_class_counts_all_pneumonia():
    labels = np.array([1, 1, 1])
    counts = get_class_counts(labels)
    assert counts["NORMAL"] == 0
    assert counts["PNEUMONIA"] == 3
