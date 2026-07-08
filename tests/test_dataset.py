"""测试 dataset 模块。"""

import numpy as np
import pytest
import torch

from dataset import CachedDataset, get_class_counts


def test_cached_dataset_len():
    images = np.random.rand(10, 3, 224, 224).astype(np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    ds = CachedDataset(images, labels)
    assert len(ds) == 10


def test_cached_dataset_getitem():
    images = np.random.rand(5, 3, 224, 224).astype(np.float32)
    labels = np.array([0, 0, 1, 1, 1])
    ds = CachedDataset(images, labels)

    img, lbl = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert lbl.item() == 0


def test_cached_dataset_with_transform():
    images = np.random.rand(3, 3, 224, 224).astype(np.float32)
    labels = np.array([0, 1, 0])

    def fake_transform(img):
        return img * 2

    ds = CachedDataset(images, labels, transform=fake_transform)
    img, _ = ds[0]
    assert torch.allclose(img, torch.from_numpy(images[0]).float() * 2)


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
