# 🫁 Pneumonia Detector

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

> **AI-powered pneumonia detection from chest X-ray images.**
>
> A production-grade desktop application and training pipeline built with PyTorch + PySide6, accelerated by a Rust preprocessing engine.

---

## Table of Contents

- [What is this?](#what-is-this)
- [⚡ Get Started](#-get-started)
- [🏗️ Architecture Overview](#️-architecture-overview)
- [🔧 CLI Reference](#-cli-reference)
- [🖥️ Desktop Application](#️-desktop-application)
- [🌟 Core Features](#-core-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [📋 Prerequisites](#-prerequisites)
- [📖 Detailed Process](#-detailed-process)
- [📁 Project Structure](#-project-structure)
- [🧪 Testing](#-testing)
- [🚀 Development](#-development)
- [📄 License](#-license)

---

## What is this?

Traditional medical image analysis requires radiologists to manually inspect thousands of X-ray images. **Pneumonia Detector** automates this with a deep-learning pipeline that classifies chest X-rays as **NORMAL** or **PNEUMONIA** in under a second.

It is designed as a **complete engineering product**, not just a research notebook:

- **Train** a ResNet-50 model with transfer learning, mixed precision, and early stopping
- **Evaluate** with ROC curves, confusion matrices, and threshold optimization
- **Detect** via an intuitive desktop GUI with real-time confidence scoring
- **Accelerate** data loading with a Rust-powered parallel preprocessing engine

The system handles the real-world challenge of **class imbalance** (1,341 NORMAL vs 3,875 PNEUMONIA cases) through weighted loss and stratified sampling.

---

## ⚡ Get Started

### 1. Install dependencies

Requires [uv](https://docs.astral.sh/uv/) (recommended) or `pip`.

```bash
# Clone the repository
git clone https://github.com/yourusername/image-detector.git
cd image-detector

# uv sync automatically creates .venv and installs all dependencies
# No manual "source .venv/bin/activate" needed — uv run handles it
uv sync

# Build the Rust extension (required for preprocessing)
uv run maturin develop
```

> [!TIP]
> `uv sync` creates and manages a virtual environment at `.venv/` automatically. `uv run` always uses this environment, so you never need to manually activate it. If you prefer the traditional workflow, you can still run `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows) and use `python` directly.

### 2. Download the dataset

```bash
uv run main.py download
```

This fetches the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle automatically.

### 3. Preprocess with Rust

```bash
uv run main.py cache
```

Generates `cache/train_images.npy` and `cache/test_images.npy` — raw JPEGs resized to `(N, 3, 256, 256)` uint8 tensors, ready for instant loading during training.

> [!NOTE]
> The Rust preprocessor uses **rayon** for parallel I/O and **Lanczos3** resampling. It skips corrupted images automatically.

### 4. Train the model

```bash
uv run main.py train
```

Training includes:
- Weighted BCE loss for class imbalance
- ReduceLROnPlateau scheduler
- Early stopping (patience = 7)
- Automatic Mixed Precision (AMP) on CUDA
- Checkpoint resume with `--resume path/to/checkpoint.pth`

### 5. Evaluate

```bash
uv run main.py evaluate
```

Generates:
- `outputs/roc_curve.png`
- `outputs/confusion_matrix.png`
- `outputs/training_history.png`
- `outputs/metrics.json`

### 6. Launch the GUI

```bash
uv run main.py gui
```

Upload a chest X-ray image and get an instant prediction with confidence score.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────┐
│              User Interaction Layer          │
│  CLI (main.py)          GUI (PySide6)        │
│  ─────────────          ───────────          │
│  train/evaluate/gui    Image upload + detect│
└────────────────────┬────────────────────────┘
                     │
┌────────────────────┴────────────────────────┐
│           Python Deep Learning Layer         │
│  model.py (ResNet-50)  →  train.py          │
│  dataset.py (CachedDataset) → evaluate.py   │
│  inference.py (single / batch)              │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────┴────────────────────────┐
│            Rust Preprocessing Layer          │
│  lib.rs (PyO3)  →  Parallel read/resize     │
│  rayon + image crate  →  .npy uint8 cache   │
└─────────────────────────────────────────────┘
```

**Key design decision:** Rust handles the **I/O-bound** preprocessing (reading 5,216 JPEGs, resizing, writing `.npy` cache). Python handles the **compute-bound** deep learning training. This separation eliminates Python's GIL bottleneck during data loading and reduces training startup time from minutes to seconds.

---

## 🔧 CLI Reference

| Command | Description |
|---------|-------------|
| `train` | Train the ResNet-50 model with early stopping and checkpointing |
| `evaluate` | Evaluate model on test set, generate ROC/CM/history plots |
| `gui` | Launch the PySide6 desktop application |
| `cache` | Run Rust preprocessing to generate `.npy` image caches |
| `download` | Download the Chest X-Ray dataset from KaggleHub |
| `dataset` | Manage registered datasets (`add`, `list`, `remove`, `set`) |

### Common flags

| Flag | Description | Example |
|------|-------------|---------|
| `--dataset-root` | Path to dataset root | `--dataset-root /path/to/chest_xray` |
| `--dataset-name` | Use a registered dataset name | `--dataset-name chest1` |
| `--cache-dir` | Override cache directory | `--cache-dir /tmp/cache` |
| `--models-dir` | Override model save directory | `--models-dir /tmp/models` |
| `--resume` | Resume from checkpoint | `--resume models/best_model.pth` |

### Environment variables

```bash
export DATASET_ROOT=/path/to/chest_xray
export DATASET_ROOTS=/path/to/dataset1,/path/to/dataset2
export CACHE_DIR=/path/to/cache
export MODELS_DIR=/path/to/models
export OUTPUTS_DIR=/path/to/outputs
```

### Dataset management

```bash
# Register a dataset
uv run main.py dataset add chest1 /path/to/chest_xray

# List registered datasets
uv run main.py dataset list

# Set default dataset
uv run main.py dataset set chest1

# Remove a dataset
uv run main.py dataset remove chest1
```

---

## 🖥️ Desktop Application

The GUI provides a **radiologist-friendly** interface for one-off diagnosis:

| Feature | Description |
|---------|-------------|
| 🖼️ Image Upload | Select `.png`, `.jpg`, or `.jpeg` X-ray images |
| 🔍 Real-time Detection | Runs inference in a background thread (no UI freeze) |
| 📊 Confidence Bar | Visual progress bar showing prediction confidence |
| 🎨 Color-coded Results | **PNEUMONIA** in red bold, **NORMAL** in green bold |
| ⚙️ Threshold Display | Shows the optimized threshold loaded from evaluation |
| 🔄 Rust Preprocess | One-click dataset preprocessing from the GUI |
| 📂 Dataset Switching | Change dataset path without restarting the app |

> [!IMPORTANT]
> All inference runs in a `QThread` worker. The UI stays responsive even during model loading or heavy preprocessing.

---

## 🌟 Core Features

### Medical Imaging Optimizations

| Feature | Implementation | Why it matters |
|---------|---------------|----------------|
| **Class imbalance handling** | `BCEWithLogitsLoss(pos_weight)` | PNEUMONIA is ~3x more common; weighting prevents model bias toward majority class |
| **Stratified split** | `sklearn.train_test_split(stratify=labels)` | Train/val sets preserve the same 25%/75% class ratio |
| **Threshold tuning** | Grid search on validation F1 | Default 0.5 is rarely optimal for medical diagnosis; we find the best operating point |
| **ImageNet normalization** | `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]` | Leverages pre-trained ResNet-50 weights effectively |

### Training Pipeline

| Feature | Implementation | Benefit |
|---------|---------------|---------|
| **Transfer learning** | ResNet-50 `IMAGENET1K_V2` | Faster convergence, better generalization on small medical datasets |
| **Mixed precision** | `torch.amp.autocast` + `GradScaler` | ~2x faster training on NVIDIA GPUs, same model quality |
| **Early stopping** | Patience = 7, monitor `val_f1` | Prevents overfitting; saves training time |
| **LR scheduling** | `ReduceLROnPlateau` | Automatically reduces LR when validation F1 plateaus |
| **Checkpoint resume** | Saves model + optimizer + scheduler + epoch | Recover from crashes or stop/resume experiments |
| **Determinism** | `set_seed(42)` | Fully reproducible results across runs |

### Data Pipeline

| Feature | Implementation | Benefit |
|---------|---------------|---------|
| **Rust preprocessing** | `rayon` parallel + `image` crate | Parallel I/O saturates disk bandwidth; no Python GIL contention |
| **Cache-first loading** | `.npy` uint8 tensors | Training starts instantly; zero per-epoch JPEG decoding |
| **Data augmentation** | RandomCrop(224), HorizontalFlip, Rotation(10°) | Increases effective training size; improves generalization |
| **Validation consistency** | CenterCrop(224) + same normalization | Ensures fair evaluation; no data leakage |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Deep Learning** | PyTorch 2.12 + torchvision | Model training, transfer learning, inference |
| **GUI** | PySide6 6.8 | Cross-platform desktop application |
| **Preprocessing** | Rust + PyO3 + maturin | High-performance parallel image I/O |
| **Data Science** | NumPy, pandas, scikit-learn | Metrics, stratified sampling, threshold tuning |
| **Visualization** | Matplotlib, Seaborn | ROC curves, confusion matrices, training history |
| **Logging** | loguru | Structured console + file logging with rotation |
| **Package Management** | uv | Fast, modern Python dependency resolution |
| **Testing** | pytest | Comprehensive unit test coverage |

---

## 📋 Prerequisites

- **OS:** Linux, macOS, or Windows
- **Python:** 3.12+
- **Rust:** 1.80+ (for building the preprocessing extension)
- **uv:** [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
- **maturin:** `pip install maturin` (for Rust/Python bridge)
- **Optional:** CUDA-capable GPU for accelerated training

---

## 📖 Detailed Process

<details>
<summary><b>Click to expand the full training and evaluation workflow</b></summary>

### STEP 1: Environment setup

```bash
# Verify Python version
python --version  # >= 3.12

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and enter the project
git clone <repo-url>
cd image-detector

# uv sync: creates .venv (if missing) + installs dependencies from pyproject.toml
uv sync

# Build Rust extension in the managed environment
uv run maturin develop
```

> [!TIP]
> Unlike `pip`, `uv` does not require you to manually create a virtual environment or activate it. `uv sync` handles `.venv` creation automatically, and every `uv run` command runs inside that environment.

### STEP 2: Dataset acquisition

The project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

```bash
# Option A: Automatic download
uv run main.py download

# Option B: Manual download
# Place the dataset at database/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/chest_xray
# Or set DATASET_ROOT to point to your local copy
```

Dataset structure:

```
chest_xray/
├── train/
│   ├── NORMAL/         # 1,341 images
│   └── PNEUMONIA/      # 3,875 images
├── test/
│   ├── NORMAL/         # 234 images
│   └── PNEUMONIA/      # 390 images
└── val/                # (optional, merged into train if present)
```

### STEP 3: Rust preprocessing

```bash
uv run main.py cache
```

This generates:
- `cache/train_images.npy` — `(N, 3, 256, 256)` uint8, merged from `train/` + `val/`
- `cache/train_labels.npy` — `(N,)` int64 labels
- `cache/test_images.npy` — `(M, 3, 256, 256)` uint8
- `cache/test_labels.npy` — `(M,)` int64 labels

> [!NOTE]
> The Rust preprocessor walks the directory tree, filters `.jpg/.jpeg/.png`, resizes to 256×256 with Lanczos3, and writes contiguous NumPy arrays. Corrupted images are skipped with a warning.

### STEP 4: Model training

```bash
# Full training (20 epochs max, early stopping)
uv run main.py train

# Resume from checkpoint
uv run main.py train --resume models/best_model.pth

# Train with a specific dataset
uv run main.py train --dataset-name chest1
```

Training monitors:
- `train_loss` — training set BCE loss
- `val_loss` / `val_f1` / `val_auc` — validation metrics

Best model is saved to `models/best_model.pth` (based on highest `val_f1`).

Training history is saved to `outputs/history.json`.

### STEP 5: Evaluation and threshold optimization

```bash
uv run main.py evaluate
```

Evaluation pipeline:
1. Load `best_model.pth`
2. Evaluate on **validation set** to find the optimal classification threshold (grid search maximizing F1)
3. Evaluate on **test set** using the optimized threshold
4. Generate plots and save metrics report

Outputs:
| File | Description |
|------|-------------|
| `outputs/roc_curve.png` | ROC curve with AUC score |
| `outputs/confusion_matrix.png` | Confusion matrix heatmap |
| `outputs/training_history.png` | Loss / F1 / AUC over epochs |
| `outputs/metrics.json` | Structured accuracy, precision, recall, F1, AUC |
| `outputs/threshold.json` | Optimal threshold for inference |

### STEP 6: GUI inference

```bash
uv run main.py gui
```

Usage:
1. Click **「更改」** to select your dataset root (if not already configured)
2. Click **「预处理」** to generate caches (if missing)
3. Click **「选择图像」** to load a chest X-ray image
4. Click **「开始检测」** to run inference
5. View the result (NORMAL / PNEUMONIA) with confidence percentage

</details>

---

## 📁 Project Structure

```
image-detector/
├── main.py                     # Project entry point (CLI dispatcher)
├── pyproject.toml              # uv project configuration
├── setup.py                    # setuptools install script
├── Cargo.toml                  # Rust package manifest
├── AGENTS.md                   # Agent operation guidelines
├── .gitignore
│
├── src/                        # Rust source
│   ├── lib.rs                  # PyO3 extension + parallel preprocessing
│   └── bin/
│       └── preprocess.rs       # Standalone Rust CLI preprocessor
│
├── src-py/                     # Python source
│   ├── main.py                 # CLI argument parsing + command dispatch
│   ├── config.py               # Global constants + path overrides
│   ├── system.py               # Cross-platform paths + dataset registry
│   ├── model.py                # ResNet-50 builder + seed setter
│   ├── dataset.py              # CachedDataset + DataLoader factory
│   ├── train.py                # Training loop with AMP/early stopping
│   ├── evaluate.py             # Evaluation + plot generation
│   ├── metrics.py              # Accuracy, precision, recall, F1, AUC
│   ├── inference.py            # Single/batch image prediction
│   ├── image_process.py        # PIL + torchvision transforms for inference
│   ├── threshold_tuner.py      # Optimal threshold search (grid)
│   ├── gui.py                  # PySide6 desktop application
│   ├── download_database.py    # KaggleHub dataset downloader
│   └── logger_config.py        # loguru configuration
│
├── tests/                      # pytest test suite
│   ├── conftest.py             # PYTHONPATH setup for tests
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_metrics.py
│   ├── test_image_process.py
│   ├── test_system.py
│   └── test_threshold_tuner.py
│
├── database/                   # Raw dataset (gitignored)
│   └── paultimothymooney/...
│
├── cache/                      # Rust-generated .npy caches (gitignored)
├── models/                     # Saved checkpoints (gitignored)
└── outputs/                    # Evaluation plots + logs (gitignored)
```

---

## 🧪 Testing

Run the full test suite:

```bash
uv run pytest
```

Run with verbose output:

```bash
uv run pytest -v
```

Coverage includes:
- ✅ Configuration constants and path overrides
- ✅ CachedDataset (len, getitem, transforms, full pipeline)
- ✅ ResNet-50 model architecture and forward pass
- ✅ Metric computation (perfect prediction, single-class edge cases)
- ✅ Image preprocessing (resize, normalize, different input sizes)
- ✅ System utilities (platform detection, path resolution, env overrides)
- ✅ Threshold tuning (save/load, missing files, multiple metrics)

---

## 🚀 Development

### Building the Rust extension

```bash
# Development mode (editable, fast rebuild)
uv run maturin develop

# Production build (wheel)
uv run maturin build
```

### Code quality

The project uses **Ruff** for linting and formatting (configured in `pyproject.toml`):

```bash
# Format
uv run ruff format .

# Lint
uv run ruff check .
```

### Adding a new dataset

```bash
uv run main.py dataset add mydata /path/to/mydata
uv run main.py dataset set mydata
uv run main.py cache
uv run main.py train
```

### Extending the model

Edit `src-py/model.py` to swap architectures:

```python
# Example: switch to ResNet-101
def build_model(pretrained=True):
    weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet101(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with 🫁 + 🦀 + 🔥
</p>
