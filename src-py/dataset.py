import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from config import (
    BATCH_SIZE,
    CACHE_DIR,
    CLASS_NAMES,
    NUM_WORKERS,
    PIN_MEMORY,
    RANDOM_SEED,
    VAL_SIZE,
)


class CachedDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def train_transform(image):
    if torch.rand(1).item() > 0.5:
        image = TF.hflip(image)
    angle = torch.empty(1).uniform_(-10, 10).item()
    image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
    return image


def val_transform(image):
    return image


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
        train_images[train_idx], train_labels[train_idx], transform=train_transform
    )
    val_dataset = CachedDataset(
        train_images[val_idx], train_labels[val_idx], transform=val_transform
    )
    test_dataset = CachedDataset(test_images, test_labels, transform=val_transform)

    train_loader = _build_dataloader(train_dataset, shuffle=True)
    val_loader = _build_dataloader(val_dataset, shuffle=False)
    test_loader = _build_dataloader(test_dataset, shuffle=False)

    return train_loader, val_loader, test_loader


def get_class_counts(labels):
    counts = np.bincount(labels.astype(int), minlength=len(CLASS_NAMES))
    return {name: int(count) for name, count in zip(CLASS_NAMES, counts)}
