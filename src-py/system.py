import multiprocessing
import os
import platform
from pathlib import Path


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_windows() -> bool:
    return platform.system() == "Windows"


def is_macos() -> bool:
    return platform.system() == "Darwin"


def get_platform_name() -> str:
    return platform.system()


def _get_home() -> Path:
    return Path.home()


def get_default_dataset_root() -> Path:
    env_root = os.getenv("DATASET_ROOT")
    if env_root:
        return Path(env_root)

    # 按优先级尝试多个候选路径，提高跨平台/跨环境的兼容性
    candidates = [
        Path("/root/autodl-tmp/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/chest_xray"),
        _get_home() / "datasets" / "chest-xray-pneumonia" / "chest_xray",
        Path("database/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/chest_xray"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def get_default_cache_dir(project_root: Path) -> Path:
    env_cache = os.getenv("CACHE_DIR")
    if env_cache:
        return Path(env_cache)
    return project_root / "cache"


def get_default_models_dir(project_root: Path) -> Path:
    env_models = os.getenv("MODELS_DIR")
    if env_models:
        return Path(env_models)
    return project_root / "models"


def get_default_outputs_dir(project_root: Path) -> Path:
    env_outputs = os.getenv("OUTPUTS_DIR")
    if env_outputs:
        return Path(env_outputs)
    return project_root / "outputs"


def get_default_kaggle_cache_dir() -> Path:
    env_kaggle = os.getenv("KAGGLEHUB_CACHE")
    if env_kaggle:
        return Path(env_kaggle)

    if is_linux():
        return Path("/root/autodl-tmp")
    return _get_home() / ".kaggle" / "hub_cache"


def normalize_path(path: Path) -> Path:
    return path.expanduser().resolve()


def get_num_workers_default() -> int:
    cpu_count = multiprocessing.cpu_count()
    return min(4, cpu_count)


def get_multiprocessing_start_method() -> str:
    if is_windows() or is_macos():
        return "spawn"
    return "fork"
