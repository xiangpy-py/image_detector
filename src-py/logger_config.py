"""全局日志配置模块。

初始化 loguru logger，使日志同时输出到控制台和文件。
在 CLI/GUI 入口（main.py）或训练脚本中导入并调用 setup_logger() 即可生效。
"""

import sys
from pathlib import Path

from loguru import logger

from config import OUTPUTS_DIR


def setup_logger(
    log_dir: Path | None = None,
    level: str = "INFO",
    console_format: str | None = None,
    file_format: str | None = None,
) -> None:
    """配置全局 logger。

    Args:
        log_dir: 日志文件存放目录，默认使用 OUTPUTS_DIR
        level: 日志级别，默认 INFO
        console_format: 控制台格式字符串，None 则使用默认
        file_format: 文件格式字符串，None 则使用默认
    """
    # 移除默认的 stderr handler，避免重复
    logger.remove()

    _console_format = console_format or (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
        "| <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    _file_format = file_format or (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} - {message}"
    )

    # 控制台输出
    logger.add(
        sys.stderr,
        level=level,
        format=_console_format,
        colorize=True,
        enqueue=True,
    )

    # 文件输出
    log_dir = log_dir or OUTPUTS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app_{time:YYYYMMDD}.log"
    logger.add(
        str(log_file),
        level=level,
        format=_file_format,
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
        encoding="utf-8",
    )
