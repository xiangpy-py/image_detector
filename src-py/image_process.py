import numpy as np
import torch
from PIL import Image

from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


def load_image(path):
    image = Image.open(path).convert("RGB")
    return image


def resize_image(image, size=IMG_SIZE):
    return image.resize((size, size), Image.Resampling.LANCZOS)


def image_to_tensor(image, size=IMG_SIZE):
    image = resize_image(image, size)
    arr = np.array(image).astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD).reshape(1, 1, 3)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
    return tensor


def preprocess_image_path(path, size=IMG_SIZE):
    image = load_image(path)
    return image_to_tensor(image, size)
