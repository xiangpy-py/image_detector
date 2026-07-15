import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from config import (
    BATCH_SIZE,
    CACHE_DIR,
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    RANDOM_SEED,
    VAL_SIZE,
)


def _get_merged_info():
    """读取多数据集合并信息文件，若不存在则返回 None。"""
    info_path = CACHE_DIR / "merged_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _is_multi_dataset():
    """检测是否处于多数据集合并模式。"""
    return _get_merged_info() is not None and len(_get_merged_info()["datasets"]) > 0


def _get_train_transforms():
    """训练数据增强：针对医学影像优化。

    关键原则：
    - 只保留几何变换，不破坏组织密度/对比度特征
    - RandomResizedCrop(scale=0.8~1.0) 避免切掉边缘病理区域
    - 旋转角度限制在 5° 内
    - 不用 ColorJitter / GaussianBlur（会破坏诊断特征）
    """
    tfms = [
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(tfms)


def _get_val_transforms():
    """验证/测试预处理：Resize(256) → CenterCrop(224) + Normalize。"""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class CachedDataset(Dataset):
    """从 Rust 预处理生成的 uint8 .npy 缓存加载数据。

    缓存图像格式: (N, 3, H, W) uint8, 其中 H=W=256。
    __getitem__ 中将 (C, H, W) 转为 PIL Image 后应用 torchvision transforms。
    """

    def __init__(self, images, labels, transform=None):
        # 保持为 numpy uint8, 不在 __init__ 中做 ToTensor/Normalize
        self.images = images
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 取出单张图像: (3, 256, 256) uint8
        image = self.images[idx]
        # (C, H, W) -> (H, W, C) for PIL
        image = np.transpose(image, (1, 2, 0))
        image = Image.fromarray(image)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_cached_data(split="train"):
    """从 Rust 预处理生成的 uint8 .npy 缓存加载数据。

    支持两种模式：
    1. 单数据集模式（兼容旧版）: 直接从 CACHE_DIR 下加载
    2. 多数据集合并模式: 从 CACHE_DIR 各子目录加载并拼接
    """
    merged_info = _get_merged_info()

    if merged_info is not None:
        # ─── 多数据集合并模式 ───
        all_images = []
        all_labels = []
        for ds in merged_info["datasets"]:
            sub_cache = Path(ds["cache_subdir"])
            images_path = sub_cache / f"{split}_images.npy"
            labels_path = sub_cache / f"{split}_labels.npy"
            if not images_path.exists() or not labels_path.exists():
                raise FileNotFoundError(
                    f"缓存文件不存在: {images_path} 或 {labels_path}，"
                    "请先运行 Rust 预处理工具生成缓存。"
                )
            images = np.load(images_path)
            labels = np.load(labels_path)

            if images.dtype != np.uint8:
                raise ValueError(
                    f"检测到旧版缓存格式 (dtype={images.dtype})，"
                    "请删除 cache 目录后重新运行缓存生成。"
                )
            all_images.append(images)
            all_labels.append(labels)

        merged_images = np.concatenate(all_images, axis=0)
        merged_labels = np.concatenate(all_labels, axis=0)
        return merged_images, merged_labels

    # ─── 单数据集模式（兼容旧版 flat cache）───
    images_path = CACHE_DIR / f"{split}_images.npy"
    labels_path = CACHE_DIR / f"{split}_labels.npy"
    if not images_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"缓存文件不存在: {images_path} 或 {labels_path}，"
            "请先运行 Rust 预处理工具生成缓存。"
        )
    images = np.load(images_path)
    labels = np.load(labels_path)

    # 格式校验：若缓存仍是旧版 float32 (已归一化) 则给出友好提示
    if images.dtype != np.uint8:
        raise ValueError(
            f"检测到旧版缓存格式 (dtype={images.dtype})，"
            "请删除 cache 目录后重新运行 `uv run python src-py/main.py cache` 生成新缓存。"
        )

    return images, labels


def _build_dataloader(dataset, shuffle=False, batch_size=None, sampler=None):
    bs = batch_size or BATCH_SIZE
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
    )


def get_dataloaders(batch_size=None, use_weighted_sampler=True):
    train_images, train_labels = load_cached_data("train")
    test_images, test_labels = load_cached_data("test")

    train_idx, val_idx = train_test_split(
        np.arange(len(train_labels)),
        test_size=VAL_SIZE,
        stratify=train_labels,
        random_state=RANDOM_SEED,
    )

    train_dataset = CachedDataset(
        train_images[train_idx], train_labels[train_idx], transform=_get_train_transforms()
    )
    val_dataset = CachedDataset(
        train_images[val_idx], train_labels[val_idx], transform=_get_val_transforms()
    )
    test_dataset = CachedDataset(test_images, test_labels, transform=_get_val_transforms())

    # ─── WeightedRandomSampler: 每个 epoch 中两类样本数平衡 ───
    sampler = None
    if use_weighted_sampler:
        train_labels_arr = train_labels[train_idx]
        counts = np.bincount(train_labels_arr.astype(int))
        # 计算每个样本的权重: 1 / 该类别样本数
        weights = 1.0 / counts[train_labels_arr.astype(int)]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=len(train_labels_arr),
            replacement=True,
        )

    train_loader = _build_dataloader(
        train_dataset, shuffle=(sampler is None), batch_size=batch_size, sampler=sampler
    )
    val_loader = _build_dataloader(val_dataset, shuffle=False, batch_size=batch_size)
    test_loader = _build_dataloader(test_dataset, shuffle=False, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def get_class_counts(labels):
    counts = np.bincount(labels.astype(int), minlength=len(CLASS_NAMES))
    return {name: int(count) for name, count in zip(CLASS_NAMES, counts)}
