import os
from pathlib import Path

import kagglehub
from loguru import logger

from logger_config import setup_logger
from system import get_default_kaggle_cache_dir


def download_dataset(cache_dir=None, dataset_root=None):
    """下载 kaggle 数据集。

    Args:
        cache_dir: kagglehub 缓存目录（兼容性参数，dataset_root 优先级更高）
        dataset_root: 数据集根目录。设置后会覆盖 cache_dir，下载到该目录。
    """
    if dataset_root is not None:
        dataset_root = Path(dataset_root)
        dataset_root.mkdir(parents=True, exist_ok=True)
        os.environ["KAGGLEHUB_CACHE"] = str(dataset_root)
    elif cache_dir is not None:
        os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)
    else:
        os.environ["KAGGLEHUB_CACHE"] = str(get_default_kaggle_cache_dir())

    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    path = Path(path)

    # 自动定位 chest_xray 数据根目录
    actual_root = _find_chest_xray_root(path)
    if actual_root:
        logger.info(f"数据集已下载: {actual_root}")
        logger.info(f"内容: {os.listdir(actual_root)}")
        return str(actual_root)

    logger.info(f"数据集路径: {path}")
    logger.info(f"内容列表: {os.listdir(path)}")
    return str(path)


def _find_chest_xray_root(path):
    """从 kagglehub 返回的路径中定位 chest_xray 数据根目录。

    kagglehub 下载后通常结构为:
        .../versions/2/chest_xray/
            ├── train/
            ├── test/
            └── val/

    Returns:
        Path | None: 包含 train/test 的目录路径，找不到则返回 None
    """
    path = Path(path)

    # 1. 当前目录下直接有 chest_xray 子目录
    chest_xray_dir = path / "chest_xray"
    if chest_xray_dir.exists():
        return chest_xray_dir

    # 2. 当前目录本身就是 chest_xray（包含 train/test）
    if (path / "train").exists() and (path / "test").exists():
        return path

    # 3. 向上搜索一级（兼容不同版本结构）
    parent = path.parent
    if (parent / "chest_xray").exists():
        return parent / "chest_xray"

    return None


if __name__ == "__main__":
    setup_logger()
    download_dataset()
