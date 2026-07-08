import os

import kagglehub
from loguru import logger

from logger_config import setup_logger
from system import get_default_kaggle_cache_dir


def download_dataset(cache_dir=None):
    if cache_dir is None:
        cache_dir = get_default_kaggle_cache_dir()

    os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)

    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

    logger.info(f"数据集路径: {path}")
    logger.info(f"内容列表: {os.listdir(path)}")
    return path


if __name__ == "__main__":
    setup_logger()
    download_dataset()
