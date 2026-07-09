"""测试 image_process 模块。"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from image_process import get_inference_transforms, preprocess_image_path


def create_test_image(path, size=(512, 512), color=(128, 128, 128)):
    """创建一张测试图像。"""
    img = Image.new("RGB", size, color)
    img.save(path)


def test_preprocess_image_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test_xray.jpg"
        # 使用白色图像，Normalize 后值会大于 1
        create_test_image(img_path, size=(512, 512), color=(255, 255, 255))

        tensor = preprocess_image_path(img_path, size=224)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        # 检查是否经过归一化（白色图像 Normalize 后应大于 1）
        assert tensor.max() > 1.0


def test_preprocess_image_path_different_size():
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test_xray.png"
        create_test_image(img_path, size=(1024, 1024))

        tensor = preprocess_image_path(img_path, size=224)
        assert tensor.shape == (3, 224, 224)


def test_get_inference_transforms():
    transform = get_inference_transforms(size=224)
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.jpg"
        create_test_image(img_path, size=(512, 512))
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)
