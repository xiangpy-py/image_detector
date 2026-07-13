from pathlib import Path

from system import (
    get_all_dataset_roots,
    get_app_data_dir,
    get_default_cache_dir,
    get_default_dataset_root,
    get_default_models_dir,
    get_default_outputs_dir,
    get_num_workers_default,
    normalize_path,
    set_app_data_dir,
)


# 项目根目录（相对于当前文件所在位置）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 全局路径（可通过 override_paths 在运行时覆盖）
DATASET_ROOT = normalize_path(get_default_dataset_root())
DATASET_ROOTS = [normalize_path(p) for p in get_all_dataset_roots()]
CACHE_DIR = normalize_path(get_default_cache_dir())
MODELS_DIR = normalize_path(get_default_models_dir())
OUTPUTS_DIR = normalize_path(get_default_outputs_dir())

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
LABEL_MAP = {"NORMAL": 0, "PNEUMONIA": 1}

IMG_SIZE = 224
CACHE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SIZE = 0.15
RANDOM_SEED = 42

NUM_WORKERS = get_num_workers_default()
PIN_MEMORY = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EARLY_STOP_PATIENCE = 7
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5

DEFAULT_THRESHOLD = 0.5


def override_paths(
    dataset_root: Path | str | None = None,
    cache_dir: Path | str | None = None,
    models_dir: Path | str | None = None,
    outputs_dir: Path | str | None = None,
    app_data_dir: Path | str | None = None,
    dataset_name: str | None = None,
) -> None:
    """在运行时覆盖全局路径配置。

    Args:
        dataset_root: 数据集根目录（直接指定路径）
        cache_dir: 缓存目录
        models_dir: 模型保存目录
        outputs_dir: 输出目录
        app_data_dir: 应用数据目录（设置后会自动推导 cache/models/outputs）
        dataset_name: 已注册数据集名称（与 dataset_root 二选一）
    """
    global DATASET_ROOT, DATASET_ROOTS, CACHE_DIR, MODELS_DIR, OUTPUTS_DIR

    if app_data_dir is not None:
        set_app_data_dir(Path(app_data_dir))
        # 重新计算依赖 app_data_dir 的路径
        if cache_dir is None:
            cache_dir = get_default_cache_dir()
        if models_dir is None:
            models_dir = get_default_models_dir()
        if outputs_dir is None:
            outputs_dir = get_default_outputs_dir()

    if dataset_name is not None:
        from system import get_dataset_path

        DATASET_ROOT = normalize_path(get_dataset_path(dataset_name))
        DATASET_ROOTS = [DATASET_ROOT]
    elif dataset_root is not None:
        DATASET_ROOT = normalize_path(Path(dataset_root))
        DATASET_ROOTS = [DATASET_ROOT]

    if cache_dir is not None:
        CACHE_DIR = normalize_path(Path(cache_dir))
    if models_dir is not None:
        MODELS_DIR = normalize_path(Path(models_dir))
    if outputs_dir is not None:
        OUTPUTS_DIR = normalize_path(Path(outputs_dir))
