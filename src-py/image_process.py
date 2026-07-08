import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_inference_transforms(size=IMG_SIZE):
    """获取与训练时验证集一致的预处理变换：Resize(256) → CenterCrop(224) → Normalize"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image_path(path, size=IMG_SIZE):
    image = Image.open(path).convert("RGB")
    transform = get_inference_transforms(size)
    return transform(image)
