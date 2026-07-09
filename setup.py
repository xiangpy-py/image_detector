#!/usr/bin/env python3
"""胸部 X 光肺炎检测系统 —— setuptools 安装脚本。

Rust 扩展需要通过 maturin 单独构建：
    pip install maturin
    maturin develop   # 开发模式（本地安装 Rust 扩展）
    maturin build     # 生产模式（生成 wheel）

纯 Python 部分安装：
    pip install -e .

完整安装（Rust + Python）：
    pip install maturin
    maturin develop
    pip install -e .
"""

from setuptools import setup

setup(
    name="image-detector",
    version="0.1.0",
    description="基于胸部X光的肺炎检测系统",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.12",
    package_dir={"": "src-py"},
    py_modules=[
        "config",
        "dataset",
        "download_database",
        "evaluate",
        "gui",
        "image_process",
        "inference",
        "logger_config",
        "main",
        "metrics",
        "model",
        "system",
        "threshold_tuner",
        "train",
    ],
    install_requires=[
        "kagglehub>=1.0.2",
        "loguru>=0.7.0",
        "matplotlib>=3.11.0",
        "opencv-python>=4.13.0.92",
        "pandas>=3.0.4",
        "pyside6>=6.8.0",
        "scikit-learn>=1.5.0",
        "seaborn>=0.13.2",
        "torch>=2.12.1",
        "torchvision>=0.27.1",
        "tqdm>=4.68.3",
    ],
    entry_points={
        "console_scripts": [
            "image-detector=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Rust",
    ],
)
