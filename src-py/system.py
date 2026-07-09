import multiprocessing
import os
import platform
from pathlib import Path


# ─── 平台检测 ───

def is_linux() -> bool:
    return platform.system() == "Linux"


def is_windows() -> bool:
    return platform.system() == "Windows"


def is_macos() -> bool:
    return platform.system() == "Darwin"


def get_platform_name() -> str:
    return platform.system()


# ─── 应用数据目录（跨平台统一）───

_APP_DATA_DIR: Path | None = None


def get_app_data_dir() -> Path:
    """获取应用数据目录，跨平台统一存放缓存、模型、输出等。

    Returns:
        Path: 应用数据根目录
    """
    global _APP_DATA_DIR
    if _APP_DATA_DIR is not None:
        return _APP_DATA_DIR

    if is_windows():
        app_data = os.getenv("LOCALAPPDATA")
        if app_data:
            _APP_DATA_DIR = Path(app_data) / "PneumoniaDetector"
        else:
            _APP_DATA_DIR = Path.home() / "AppData" / "Local" / "PneumoniaDetector"
    elif is_macos():
        _APP_DATA_DIR = Path.home() / "Library" / "Application Support" / "PneumoniaDetector"
    else:
        _APP_DATA_DIR = Path.home() / ".pneumonia-detector"

    _APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _APP_DATA_DIR


def set_app_data_dir(path: Path | str) -> None:
    """覆盖应用数据目录（用于 GUI 手动选择）。"""
    global _APP_DATA_DIR
    _APP_DATA_DIR = Path(path).expanduser().resolve()
    _APP_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ─── 路径工具 ───

def _get_home() -> Path:
    return Path.home()


def normalize_path(path: Path) -> Path:
    return path.expanduser().resolve()


# ─── 默认路径 ───

def get_default_dataset_root() -> Path:
    """返回数据集根目录。按优先级：
    1. 环境变量 DATASET_ROOT
    2. 已存在的项目 data/ 目录
    3. 应用数据目录下的 datasets/
    4. 用户主目录下的 datasets/
    """
    env_root = os.getenv("DATASET_ROOT")
    if env_root:
        return Path(env_root)

    candidates = [
        Path("database/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/chest_xray"),
        _get_home() / "datasets" / "chest-xray-pneumonia" / "chest_xray",
        get_app_data_dir() / "datasets" / "chest-xray-pneumonia" / "chest_xray",
    ]

    for path in candidates:
        if path.exists():
            return path

    # 都不存在时返回应用数据目录（后续由 GUI 或 CLI 引导用户选择）
    return candidates[2]


def get_default_cache_dir() -> Path:
    env_cache = os.getenv("CACHE_DIR")
    if env_cache:
        return Path(env_cache)
    return get_app_data_dir() / "cache"


def get_default_models_dir() -> Path:
    env_models = os.getenv("MODELS_DIR")
    if env_models:
        return Path(env_models)
    return get_app_data_dir() / "models"


def get_default_outputs_dir() -> Path:
    env_outputs = os.getenv("OUTPUTS_DIR")
    if env_outputs:
        return Path(env_outputs)
    return get_app_data_dir() / "outputs"


def get_default_kaggle_cache_dir() -> Path:
    env_kaggle = os.getenv("KAGGLEHUB_CACHE")
    if env_kaggle:
        return Path(env_kaggle)
    return get_app_data_dir() / "kaggle_cache"


# ─── 多进程 ───

def get_num_workers_default() -> int:
    cpu_count = multiprocessing.cpu_count()
    return min(4, cpu_count)


def get_multiprocessing_start_method() -> str:
    if is_windows() or is_macos():
        return "spawn"
    return "fork"
