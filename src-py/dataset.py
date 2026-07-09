import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
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


def _get_train_transforms():
    """训练数据增强：在 uint8 图像上先做空间变换，再 ToTensor + Normalize。"""
    return transforms.Compose(
        [
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _get_val_transforms():
    """验证/测试预处理：CenterCrop + ToTensor + Normalize。"""
    return transforms.Compose(
        [
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


def _build_dataloader(dataset, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
    )


def get_dataloaders():
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

    train_loader = _build_dataloader(train_dataset, shuffle=True)
    val_loader = _build_dataloader(val_dataset, shuffle=False)
    test_loader = _build_dataloader(test_dataset, shuffle=False)

    return train_loader, val_loader, test_loader


def get_class_counts(labels):
    counts = np.bincount(labels.astype(int), minlength=len(CLASS_NAMES))
    return {name: int(count) for name, count in zip(CLASS_NAMES, counts)}
