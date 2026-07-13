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


# ─── 数据集注册表 ───

import json

_DATASET_REGISTRY: dict | None = None


def _get_dataset_registry_path() -> Path:
    """返回数据集注册表 JSON 文件路径。"""
    return get_app_data_dir() / "datasets.json"


def load_dataset_registry() -> dict:
    """加载数据集注册表。

    Returns:
        dict: {"datasets": {name: {"path": str, "added_at": str}}, "active": str}
    """
    global _DATASET_REGISTRY
    if _DATASET_REGISTRY is not None:
        return _DATASET_REGISTRY

    reg_path = _get_dataset_registry_path()
    if reg_path.exists():
        with open(reg_path, "r", encoding="utf-8") as f:
            _DATASET_REGISTRY = json.load(f)
    else:
        _DATASET_REGISTRY = {"datasets": {}, "active": None}
    return _DATASET_REGISTRY


def save_dataset_registry(data: dict) -> None:
    """保存数据集注册表到磁盘。"""
    global _DATASET_REGISTRY
    _DATASET_REGISTRY = data
    reg_path = _get_dataset_registry_path()
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def add_dataset(name: str, path: Path | str) -> None:
    """注册一个新数据集。

    Args:
        name: 数据集名称
        path: 数据集根目录路径
    """
    reg = load_dataset_registry()
    reg["datasets"][name] = {
        "path": str(normalize_path(Path(path))),
        "added_at": str(Path.home()),  # 简单标记
    }
    if reg["active"] is None:
        reg["active"] = name
    save_dataset_registry(reg)


def remove_dataset(name: str) -> None:
    """移除已注册的数据集。

    Args:
        name: 数据集名称

    Raises:
        KeyError: 数据集不存在
    """
    reg = load_dataset_registry()
    if name not in reg["datasets"]:
        raise KeyError(f"数据集 '{name}' 未注册")
    del reg["datasets"][name]
    if reg["active"] == name:
        reg["active"] = next(iter(reg["datasets"]), None)
    save_dataset_registry(reg)


def list_datasets() -> dict:
    """列出所有已注册的数据集。

    Returns:
        dict: {name: {"path": str, ...}}
    """
    return load_dataset_registry()["datasets"]


def get_dataset_path(name: str) -> Path:
    """获取指定数据集的路径。

    Args:
        name: 数据集名称

    Returns:
        Path: 数据集根目录

    Raises:
        KeyError: 数据集不存在
    """
    reg = load_dataset_registry()
    if name not in reg["datasets"]:
        raise KeyError(f"数据集 '{name}' 未注册，可用: {list(reg['datasets'].keys())}")
    return normalize_path(Path(reg["datasets"][name]["path"]))


def set_active_dataset(name: str) -> None:
    """设置默认数据集。

    Args:
        name: 数据集名称

    Raises:
        KeyError: 数据集不存在
    """
    reg = load_dataset_registry()
    if name not in reg["datasets"]:
        raise KeyError(f"数据集 '{name}' 未注册")
    reg["active"] = name
    save_dataset_registry(reg)


def get_active_dataset() -> str | None:
    """获取当前默认数据集名称。"""
    return load_dataset_registry()["active"]


# ─── 路径工具 ───


def _get_home() -> Path:
    return Path.home()


def normalize_path(path: Path) -> Path:
    return path.expanduser().resolve()


# ─── 默认路径 ───


def get_default_dataset_root() -> Path:
    """返回数据集根目录。按优先级：
    1. 环境变量 DATASET_ROOT（单个路径）
    2. 环境变量 DATASET_ROOTS（逗号分隔的多个路径，取第一个）
    3. 注册表中设置的默认数据集
    4. 已存在的项目 data/ 目录
    5. 应用数据目录下的 datasets/
    6. 用户主目录下的 datasets/
    """
    # 1. 环境变量 DATASET_ROOT
    env_root = os.getenv("DATASET_ROOT")
    if env_root:
        return Path(env_root)

    # 2. 环境变量 DATASET_ROOTS（逗号分隔，取第一个）
    env_roots = os.getenv("DATASET_ROOTS")
    if env_roots:
        first_path = env_roots.split(",")[0].strip()
        if first_path:
            return Path(first_path)

    # 3. 注册表中的默认数据集
    reg = load_dataset_registry()
    active = reg.get("active")
    if active and active in reg.get("datasets", {}):
        return normalize_path(Path(reg["datasets"][active]["path"]))

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


def get_all_dataset_roots() -> list[Path]:
    """获取所有配置的数据集根目录。

    按优先级：
    1. 环境变量 DATASET_ROOTS（逗号分隔，返回所有）
    2. 环境变量 DATASET_ROOT（单个）
    3. 注册表中的所有数据集
    4. 默认探测路径

    Returns:
        list[Path]: 数据集根目录列表
    """
    # 1. DATASET_ROOTS
    env_roots = os.getenv("DATASET_ROOTS")
    if env_roots:
        paths = [Path(p.strip()) for p in env_roots.split(",") if p.strip()]
        if paths:
            return paths

    # 2. DATASET_ROOT
    env_root = os.getenv("DATASET_ROOT")
    if env_root:
        return [Path(env_root)]

    # 3. 注册表中的所有数据集
    reg = load_dataset_registry()
    datasets = reg.get("datasets", {})
    if datasets:
        return [normalize_path(Path(info["path"])) for info in datasets.values()]

    # 4. 默认路径
    return [get_default_dataset_root()]

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
