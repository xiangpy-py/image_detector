from pathlib import Path

from system import (
    get_default_cache_dir,
    get_default_dataset_root,
    get_default_models_dir,
    get_default_outputs_dir,
    get_num_workers_default,
    normalize_path,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = normalize_path(get_default_dataset_root())
CACHE_DIR = normalize_path(get_default_cache_dir(PROJECT_ROOT))
MODELS_DIR = normalize_path(get_default_models_dir(PROJECT_ROOT))
OUTPUTS_DIR = normalize_path(get_default_outputs_dir(PROJECT_ROOT))

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
LABEL_MAP = {"NORMAL": 0, "PNEUMONIA": 1}

IMG_SIZE = 224
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
) -> None:
    global DATASET_ROOT, CACHE_DIR, MODELS_DIR, OUTPUTS_DIR

    if dataset_root is not None:
        DATASET_ROOT = normalize_path(Path(dataset_root))
    if cache_dir is not None:
        CACHE_DIR = normalize_path(Path(cache_dir))
    if models_dir is not None:
        MODELS_DIR = normalize_path(Path(models_dir))
    if outputs_dir is not None:
        OUTPUTS_DIR = normalize_path(Path(outputs_dir))
