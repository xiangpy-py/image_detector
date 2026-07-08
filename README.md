# 基于胸部 X 光图像的肺炎检测系统

本项目基于 Kaggle 的 **Chest X-Ray Images (Pneumonia)** 数据集，使用 PyTorch + ResNet-50 构建二分类模型，并配以 **PySide6** 桌面应用实现单张 X 光图像的肺炎检测。同时引入 Rust 预处理工具（支持并行化），将 JPEG 图像批量转换为 `.npy` 缓存以加速训练。

## 项目结构

```text
/root/image_detector
├── src-py/                 # Python 核心代码
│   ├── config.py           # 路径与超参数配置
│   ├── dataset.py          # 数据集加载、分层划分、DataLoader
│   ├── model.py            # ResNet-50 二分类模型
│   ├── train.py            # 训练脚本（含 AMP 混合精度、早停）
│   ├── evaluate.py         # 评估与可视化
│   ├── metrics.py          # 评估指标与损失函数（训练/评估共用）
│   ├── inference.py        # 单图/批量推理
│   ├── gui.py              # PySide6 桌面应用
│   ├── image_process.py    # 图像预处理辅助函数
│   ├── download_database.py# 数据集下载（KaggleHub）
│   ├── logger_config.py    # 全局日志配置（loguru）
│   ├── threshold_tuner.py  # 验证集最优阈值搜索
│   ├── system.py           # 跨平台路径与多进程抽象
│   └── main.py             # CLI 入口
├── src/bin/preprocess.rs   # Rust 图像预处理缓存工具（rayon 并行化）
├── tests/                  # pytest 单元测试
├── models/                 # 保存训练好的模型（best_model.pth）
├── cache/                  # Rust 生成的 .npy 缓存
├── outputs/                # 评估图表、metrics.json、日志
├── pyproject.toml          # uv 依赖配置（含 ruff/black/loguru）
└── Cargo.toml              # Rust 依赖配置（含 rayon）
```

## 环境要求

- Python 3.12
- uv（Python 包管理）
- Rust + Cargo（可选，用于生成图像缓存；启用 rayon 并行化）
- NVIDIA GPU（推荐，已测试 RTX 3080 Ti + CUDA 12.x）

## 安装依赖

```bash
cd /root/image_detector
uv sync
```

主要依赖：PyTorch、torchvision、**PySide6**、matplotlib、seaborn、scikit-learn、numpy、pandas、**loguru**、tqdm。

## 数据准备

数据集默认路径按优先级自动探测：

1. 环境变量 `DATASET_ROOT`
2. `/root/autodl-tmp/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/chest_xray`
3. `~/datasets/chest-xray-pneumonia/chest_xray`
4. `database/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/chest_xray`

若数据不存在，可运行：

```bash
uv run python src-py/main.py download
```

或在任意位置设置环境变量：

```bash
export DATASET_ROOT=/your/custom/dataset/path
```

## 快速开始

### 1. 生成图像缓存（Rust）

```bash
cargo run --release --bin preprocess -- --root <数据集路径> --out cache/
```

或在项目根目录运行：

```bash
uv run python src-py/main.py cache
```

缓存文件：

- `cache/train_images.npy` / `cache/train_labels.npy`（已合并官方 train + val）
- `cache/test_images.npy` / `cache/test_labels.npy`

> Rust 预处理工具使用 **rayon** 多线程并行处理图像，单张图像损坏不会导致整个程序崩溃。

### 2. 训练模型

```bash
uv run python src-py/main.py train
```

训练完成后，最佳模型保存至 `models/best_model.pth`，训练历史保存至 `outputs/history.json`。

> 日志同时输出到控制台和 `outputs/app_YYYYMMDD.log`。

### 3. 评估模型

```bash
uv run python src-py/main.py evaluate
```

评估输出：

- `outputs/metrics.json`：验证集与测试集指标
- `outputs/roc_curve.png`：ROC 曲线
- `outputs/confusion_matrix.png`：混淆矩阵
- `outputs/training_history.png`：训练历史曲线
- `outputs/threshold.json`：验证集搜索到的最优决策阈值

### 4. 启动 GUI

```bash
uv run python src-py/main.py gui
```

在界面中选择胸部 X 光图像，点击「开始检测」，即可查看预测结果与置信度。

### 5. 运行单元测试

```bash
uv run pytest tests/ -v
```

测试覆盖：metrics、threshold_tuner、dataset、model、image_process、system、config 等核心模块。

## 跨平台注意事项

| 平台 | 说明 |
|------|------|
| **Linux** | 多进程使用 `fork`；默认路径优先探测 AutoDL 环境 |
| **Windows** | 多进程强制使用 `spawn`；Rust 二进制自动检测 `.exe` 后缀 |
| **macOS** | 多进程强制使用 `spawn`，避免 fork 导致的 CUDA/BLAS 崩溃 |

## 模型设计

- 基础网络：`torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`
- 输出层：替换 `fc` 为 `nn.Linear(2048, 1)`，输出单 logit
- 损失函数：`BCEWithLogitsLoss(pos_weight)`，使用 `normal_count / pneumonia_count` 作为正例权重，缓解类别不平衡
- 优化器：AdamW，学习率 1e-4，weight decay 1e-4
- 学习率调度：ReduceLROnPlateau（监控验证 F1）
- 早停： patience=7
- 混合精度：自动检测 CUDA 并启用 AMP（autocast + GradScaler）
- Checkpoint：保存模型、优化器、scheduler 完整状态

## 数据划分

官方验证集仅 16 张，本项目合并官方 `train` + `val`（共 5216+ 张），使用 `sklearn.model_selection.train_test_split` 按 85%/15% 分层抽样得到新的训练集与验证集，官方 `test` 集作为最终测试集。

## 预处理策略

- **训练**（由 Rust 预处理工具完成）：`Resize(224) → Normalize(ImageNet)`，保存为 `.npy` 缓存
- **验证/测试/GUI**：`Resize(256) → CenterCrop(224) → Normalize(ImageNet)`
- **数据增强**（Python DataLoader 中）：RandomHorizontalFlip、RandomRotation(±10°)

## 日志系统

项目使用 **loguru** 替代 `print`，提供：

- 彩色控制台输出（时间、级别、模块名、函数名、行号）
- 按天滚动的日志文件（`outputs/app_YYYYMMDD.log`，10MB 自动轮转，保留 7 天）
- 线程安全，适用于 DataLoader 多 worker 环境

## 代码风格

项目配置 **ruff**（lint）和 **black**（format）：

```bash
uv run ruff check src-py/         # 代码检查
uv run ruff check --fix src-py/   # 自动修复
uv run black src-py/              # 代码格式化
```

## 性能指标

在测试集上评估：Accuracy、Precision、Recall、F1-Score、AUC-ROC，并绘制 ROC 曲线与混淆矩阵。
