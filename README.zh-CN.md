# 🫁 Pneumonia Detector

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

> **基于胸部 X 光图像的 AI 肺炎检测系统。**
>
> 一个生产级的桌面应用程序和训练流水线，基于 PyTorch + PySide6 构建，由 Rust 预处理引擎加速。

---

## 目录

- [这是什么？](#这是什么)
- [🚀 如何使用](#-如何使用)
- [🏗️ 架构概览](#️-架构概览)
- [🔧 CLI 参考](#-cli-参考)
- [🖥️ 桌面应用程序](#️-桌面应用程序)
- [🌟 核心特性](#-核心特性)
- [🛠️ 技术栈](#️-技术栈)
- [📋 前置条件](#-前置条件)
- [📁 项目结构](#-项目结构)
- [🧪 测试](#-测试)
- [💻 开发](#-开发)
- [📄 许可证](#-许可证)

---

## 这是什么？

传统的医学影像分析需要放射科医生手动检查数千张 X 光片。**Pneumonia Detector** 通过深度学习流水线实现自动化分类，将胸部 X 光片判定为 **NORMAL（正常）** 或 **PNEUMONIA（肺炎）**，耗时不到一秒。

它被设计为一个**完整的工程化产品**，而非仅仅是一个研究笔记本：

- **训练** —— 默认 DenseNet121 模型（可配置：ResNet-50 + SE 注意力、EfficientNet-B0/B4、ConvNeXt-Tiny），支持迁移学习、混合精度、EMA 和早停
- **评估** —— ROC 曲线、混淆矩阵和阈值优化
- **检测** —— 直观的桌面 GUI，实时显示置信度分数
- **加速** —— Rust 驱动的并行预处理引擎

系统处理现实世界中的**类别不平衡**挑战（1,341 例 NORMAL vs 3,875 例 PNEUMONIA），通过加权损失和分层采样来解决。

---

## 🚀 如何使用

本节介绍从安装到推理的完整工作流。

### 步骤 1：安装依赖

需要 [uv](https://docs.astral.sh/uv/)（推荐）或 `pip`。

```bash
# 验证 Python 版本
python --version  # >= 3.12

# 克隆仓库
git clone https://github.com/yourusername/image-detector.git
cd image-detector

# uv sync 会自动创建 .venv 并安装所有依赖
uv sync

# 构建 Rust 扩展（预处理必需）
# 首次构建可能需要数分钟；后续构建为增量编译。
uv run maturin develop
```

> [!TIP]
> `uv sync` 会自动在项目根目录创建并管理 `.venv/` 虚拟环境。`uv run` 始终使用该环境中的 Python，所以你不需要手动激活。如果你习惯传统工作流，也可以手动运行 `source .venv/bin/activate`（Linux/macOS）或 `.venv\Scripts\activate`（Windows），然后直接使用 `python`。

### 步骤 2：下载数据集

```bash
# 从 KaggleHub 自动下载
uv run main.py download
```

自动获取 [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) 数据集。你也可以将手动下载的数据放到 `database/` 目录下，或设置 `DATASET_ROOT` 指向本地副本。

数据集目录结构：

```
chest_xray/
├── train/
│   ├── NORMAL/         # 1,341 张图像
│   └── PNEUMONIA/      # 3,875 张图像
├── test/
│   ├── NORMAL/         # 234 张图像
│   └── PNEUMONIA/      # 390 张图像
└── val/                #（可选，如存在则合并到 train）
```

### 步骤 3：使用 Rust 预处理

```bash
uv run main.py cache
```

生成以下文件：
- `cache/train_images.npy` —— `(N, 3, 256, 256)` uint8，合并自 `train/` + `val/`
- `cache/train_labels.npy` —— `(N,)` int64 标签
- `cache/test_images.npy` —— `(M, 3, 256, 256)` uint8
- `cache/test_labels.npy` —— `(M,)` int64 标签

> [!NOTE]
> Rust 预处理器遍历目录树，过滤 `.jpg/.jpeg/.png`，使用 Lanczos3 缩放到 256×256，然后写入连续的 NumPy 数组。损坏的图像会被跳过并发出警告。

### 步骤 4：训练模型

```bash
# 完整训练（最多 30 轮，早停）
uv run main.py train

# 从检查点恢复
uv run main.py train --resume models/20260715_1401_best_model.pth

# 使用指定的已注册数据集训练
uv run main.py train --dataset-name chest1
```

训练包含以下特性：
- 标签平滑 + 加权 BCE 损失处理类别不平衡
- `WeightedRandomSampler` 实现均衡 mini-batch
- 两阶段迁移学习：先冻结 backbone，再解冻并微调
- Warmup + Cosine 退火 + `ReduceLROnPlateau` 学习率调度
- EMA（指数移动平均）稳定推理
- 梯度裁剪
- 早停（patience = 10，监控 `val_f1`）
- CUDA 上的自动混合精度（AMP）

最佳模型以时间戳命名保存到 `models/`，例如：
- `models/YYYYMMDD_HHMM_best_model.pth` —— 最高 `val_f1`
- `models/YYYYMMDD_HHMM_best_auc_model.pth` —— 最高 `val_auc`
- `models/YYYYMMDD_HHMM_last_model.pth` —— 最后一轮检查点

训练历史保存到 `outputs/history.json`。

### 步骤 5：评估与阈值优化

```bash
uv run main.py evaluate
```

`evaluate` 子命令会自动加载 `models/` 目录下时间戳最新的 `*_best_model.pth` 检查点，然后：

1. 在**验证集**上评估，通过网格搜索找到最优分类阈值（以 F1 最大化为目标）。
2. 使用优化后的阈值在**测试集**上评估。
3. 生成图表并保存结构化指标报告。

生成文件：

| 文件 | 说明 |
|------|------|
| `outputs/roc_curve.png` | ROC 曲线及 AUC 分数 |
| `outputs/confusion_matrix.png` | 混淆矩阵热力图 |
| `outputs/training_history.png` | 各轮次的 Loss / F1 / AUC |
| `outputs/metrics.json` | 准确率、精确率、召回率、F1、AUC |
| `outputs/threshold.json` | 推理用的最佳阈值 |

### 步骤 6：启动 GUI

```bash
uv run main.py gui
```

GUI 使用步骤：
1. 点击 **「更改」** 选择数据集根目录（如未配置）。
2. 点击 **「预处理」** 生成缓存（如缺失）。
3. 点击 **「选择图像」** 加载胸部 X 光图像。
4. 点击 **「开始检测」** 运行推理。
5. 查看结果（**NORMAL** / **PNEUMONIA**）及置信度百分比。

---

## 🏗️ 架构概览

```
┌─────────────────────────────────────────────┐
│              用户交互层                       │
│  CLI (main.py)          GUI (PySide6)        │
│  ─────────────          ───────────          │
│  train/evaluate/gui    图像上传 + 检测       │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────┴────────────────────────┐
│           Python 深度学习层                  │
│  model.py (DenseNet121*) →  train.py        │
│  dataset.py (CachedDataset) → evaluate.py   │
│  inference.py (单张 / 批量推理)              │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────┴────────────────────────┐
│            Rust 预处理层                     │
│  lib.rs (PyO3)  →  processor.rs             │
│  dataset.rs + image.rs + cache.rs           │
│  rayon + image crate  →  .npy uint8 缓存    │
└─────────────────────────────────────────────┘

* 模型架构可在 src-py/config.py 中配置
```

**关键设计决策：** Rust 处理 **I/O 密集型** 的预处理（读取约 5,856 张 JPEG、缩放、写入 `.npy` 缓存）。Python 处理 **计算密集型** 的深度学习训练。这种分离消除了 Python GIL 在数据加载时的瓶颈，将训练启动时间从分钟级缩短到秒级。

---

## 🔧 CLI 参考

| 命令 | 说明 |
|------|------|
| `train` | 训练模型（默认 DenseNet121），支持早停和检查点保存 |
| `evaluate` | 在测试集上评估模型，生成 ROC / 混淆矩阵 / 历史图 |
| `gui` | 启动 PySide6 桌面应用程序 |
| `cache` | 运行 Rust 预处理，生成 `.npy` 图像缓存 |
| `download` | 从 KaggleHub 下载胸部 X 光数据集 |
| `dataset` | 管理已注册的数据集（`add`、`list`、`remove`、`set`） |

### 常用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--dataset-root` | 数据集根目录路径 | `--dataset-root /path/to/chest_xray` |
| `--dataset-name` | 使用已注册的数据集名称 | `--dataset-name chest1` |
| `--cache-dir` | 覆盖缓存目录 | `--cache-dir /tmp/cache` |
| `--models-dir` | 覆盖模型保存目录 | `--models-dir /tmp/models` |
| `--resume` | 从检查点恢复 | `--resume models/20260715_1401_best_model.pth` |

### 环境变量

```bash
export DATASET_ROOT=/path/to/chest_xray
export DATASET_ROOTS=/path/to/dataset1,/path/to/dataset2
export CACHE_DIR=/path/to/cache
export MODELS_DIR=/path/to/models
export OUTPUTS_DIR=/path/to/outputs
```

### 数据集管理

```bash
# 注册数据集
uv run main.py dataset add chest1 /path/to/chest_xray

# 列出已注册数据集
uv run main.py dataset list

# 设置默认数据集
uv run main.py dataset set chest1

# 移除数据集
uv run main.py dataset remove chest1
```

---

## 🖥️ 桌面应用程序

GUI 提供了一个**对放射科医生友好**的单次诊断界面：

| 特性 | 说明 |
|------|------|
| 🖼️ 图像上传 | 选择 `.png`、`.jpg` 或 `.jpeg` 格式的 X 光图像 |
| 🔍 实时检测 | 在后台线程中运行推理（UI 不卡顿） |
| 📊 置信度条 | 可视化进度条显示预测置信度 |
| 🎨 颜色编码结果 | **PNEUMONIA** 红色加粗，**NORMAL** 绿色加粗 |
| ⚙️ 阈值显示 | 显示从评估加载的优化阈值 |
| 🔄 Rust 预处理 | 从 GUI 一键执行数据集预处理 |
| 📂 数据集切换 | 无需重启应用即可更改数据集路径 |

> [!IMPORTANT]
> 所有推理都在 `QThread` 工作线程中运行。UI 在模型加载或繁重预处理期间保持响应。

---

## 🌟 核心特性

### 医学影像优化

| 特性 | 实现方式 | 为什么重要 |
|------|---------|-----------|
| **类别不平衡处理** | `LabelSmoothingBCEWithLogitsLoss(pos_weight)` + `WeightedRandomSampler` | PNEUMONIA 约是 NORMAL 的 3 倍；加权与均衡采样防止模型偏向多数类 |
| **分层划分** | `train_test_split(stratify=labels)` | 训练 / 验证集保持类别比例；原始 val/ 合并到 train |
| **阈值调优** | 在验证集 F1 上进行网格搜索 | 默认 0.5 对医学诊断 rarely 最优；我们找到最佳工作点 |
| **ImageNet 归一化** | `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]` | 有效利用预训练 ImageNet 权重 |

### 训练流水线

| 特性 | 实现方式 | 优势 |
|------|---------|------|
| **迁移学习** | DenseNet121 `IMAGENET1K_V1`（可配置） | 在小型医学数据集上收敛更快、泛化更好 |
| **混合精度** | `torch.amp.autocast` + `GradScaler` | 在 NVIDIA GPU 上训练速度提升约 2 倍，模型质量不变 |
| **两阶段微调** | 先冻结 backbone 训练 N 轮，再解冻以更低学习率微调 | 稳定预训练特征后再做全局调整 |
| **Warmup + Cosine** | 线性预热，然后余弦退火 | 避免破坏预训练权重；平滑学习率衰减 |
| **早停** | Patience = 10，监控 `val_f1` | 防止过拟合；节省训练时间 |
| **学习率调度** | `ReduceLROnPlateau` | 验证 F1 停滞时自动降低学习率 |
| **EMA** | 模型权重的指数移动平均 | 推理更稳定，泛化更好 |
| **梯度裁剪** | `max_norm=1.0` | 微调时防止梯度爆炸 |
| **检查点恢复** | 保存 model + optimizer + scheduler + epoch + EMA | 从崩溃中恢复，或随时中断 / 继续实验 |
| **确定性** | `set_seed(42)` | 每次运行结果完全可复现 |

### 数据流水线

| 特性 | 实现方式 | 优势 |
|------|---------|------|
| **Rust 预处理** | `rayon` 并行 + `image` crate | 并行 I/O 充分压榨磁盘带宽；无 Python GIL 竞争 |
| **缓存优先加载** | `.npy` uint8 张量 `(N, 3, 256, 256)` | 训练瞬间启动；每轮训练零 JPEG 解码开销 |
| **数据增强** | RandomResizedCrop(224, scale=0.8-1.0)、HorizontalFlip、Rotation(5°)、RandomAffine | 仅几何变换，保留组织密度；提升泛化能力 |
| **验证一致性** | Resize(256) → CenterCrop(224) + 相同归一化 | 确保公平评估；无数据泄露 |
| **医学安全增强** | 不使用 ColorJitter / GaussianBlur | 保护 X 光诊断所需的密度 / 对比度信息 |

---

## 🛠️ 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| **深度学习** | PyTorch 2.12 + torchvision | 模型训练、迁移学习、推理 |
| **GUI** | PySide6 6.8 | 跨平台桌面应用 |
| **预处理** | Rust + PyO3 + maturin | 高性能并行图像 I/O |
| **数据科学** | NumPy、pandas、scikit-learn | 指标、分层采样、阈值调优 |
| **可视化** | Matplotlib、Seaborn | ROC 曲线、混淆矩阵、训练历史 |
| **日志** | loguru | 结构化控制台 + 文件日志，支持轮转 |
| **包管理** | uv | 快速、现代化的 Python 依赖解析 |
| **测试** | pytest | 全面的单元测试覆盖 |

---

## 📋 前置条件

- **操作系统：** Linux、macOS 或 Windows
- **Python：** 3.12+
- **Rust：** 1.80+（用于构建预处理扩展）
- **uv：** [安装 uv](https://docs.astral.sh/uv/getting-started/installation/)
- **可选：** 支持 CUDA 的 GPU，用于加速训练（CPU 训练也可行，但速度慢很多）

---

## 📁 项目结构

```
image-detector/
├── main.py                     # 项目入口（CLI 分发器）
├── pyproject.toml              # uv 项目配置
├── setup.py                    # setuptools 安装脚本
├── Cargo.toml                  # Rust 包配置
├── AGENTS.md                   # Agent 操作指南
├── .gitignore
│
├── src/                        # Rust 源码
│   ├── lib.rs                  # PyO3 扩展入口
│   ├── processor.rs            # 并行图像处理流水线
│   ├── dataset.rs              # 数据集目录扫描
│   ├── image.rs                # 图像加载 + Lanczos3 缩放
│   ├── cache.rs                # NumPy 缓存写入
│   └── bin/
│       └── preprocess.rs       # 独立 Rust CLI 预处理器
│
├── src-py/                     # Python 源码
│   ├── main.py                 # CLI 参数解析 + 命令分发
│   ├── config.py               # 全局常量 + 路径覆盖
│   ├── system.py               # 跨平台路径 + 数据集注册表
│   ├── model.py                # 模型构建器（默认 DenseNet121）+ 种子设置
│   ├── dataset.py              # CachedDataset + DataLoader 工厂
│   ├── train.py                # 带 AMP / EMA / 早停的训练循环
│   ├── evaluate.py             # 评估 + 图表生成
│   ├── metrics.py              # 准确率、精确率、召回率、F1、AUC、损失函数
│   ├── inference.py            # 单张 / 批量图像预测
│   ├── image_process.py        # 推理用的 PIL + torchvision 变换
│   ├── threshold_tuner.py      # 最佳阈值搜索（网格）
│   ├── gui.py                  # PySide6 桌面应用
│   ├── download_database.py    # KaggleHub 数据集下载器
│   └── logger_config.py        # loguru 配置
│
├── tests/                      # pytest 测试套件
│   ├── conftest.py             # 测试的 PYTHONPATH 设置
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_metrics.py
│   ├── test_image_process.py
│   ├── test_system.py
│   └── test_threshold_tuner.py
│
├── database/                   # 原始数据集（gitignored）
│   └── paultimothymooney/...
│
├── cache/                      # Rust 生成的 .npy 缓存（gitignored）
├── models/                     # 保存的检查点（gitignored）
├── outputs/                    # 评估图表 + 日志（gitignored）
├── datasets.json               # 数据集注册表（由 CLI/GUI 创建）
└── 报告/                        # 实验报告目录（gitignored）
```

---

## 🧪 测试

运行完整测试套件：

```bash
uv run pytest
```

带详细输出运行：

```bash
uv run pytest -v
```

覆盖范围包括：
- ✅ 配置常量和路径覆盖
- ✅ CachedDataset（len、getitem、变换、完整流水线）
- ✅ DenseNet121 / ResNet-50 / EfficientNet / ConvNeXt 模型架构和前向传播
- ✅ 指标计算（完美预测、单类边界情况）
- ✅ 图像预处理（缩放、归一化、不同输入尺寸）
- ✅ 系统工具（平台检测、路径解析、环境变量覆盖）
- ✅ 阈值调优（保存 / 加载、缺失文件、多指标）

---

## 💻 开发

### 构建 Rust 扩展

```bash
# 开发模式（可编辑，快速重建）
uv run maturin develop

# 生产构建（wheel）
uv run maturin build
```

### 代码质量

项目使用 **Ruff** 进行格式化和检查（配置在 `pyproject.toml` 中）：

```bash
# 格式化
uv run ruff format .

# 检查
uv run ruff check .
```

### 添加新数据集

```bash
uv run main.py dataset add mydata /path/to/mydata
uv run main.py dataset set mydata
uv run main.py cache
uv run main.py train
```

### 扩展模型

编辑 `src-py/config.py` 以更换架构：

```python
# 通过修改 src-py/config.py 中的 MODEL_ARCH 切换架构
MODEL_ARCH = "resnet50"  # "densenet121" | "resnet50" | "efficientnet_b0" | "efficientnet_b4" | "convnext_tiny"
```

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

<p align="center">
  用 🫁 + 🦀 + 🔥 构建
</p>
