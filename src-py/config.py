from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path(
    "/root/autodl-tmp/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/chest_xray"
)
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
LABEL_MAP = {"NORMAL": 0, "PNEUMONIA": 1}

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SIZE = 0.15
RANDOM_SEED = 42

NUM_WORKERS = 4
PIN_MEMORY = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
