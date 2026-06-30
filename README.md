# 基于胸部 X 光图像的肺炎检测系统

本项目基于 Kaggle 的 **Chest X-Ray Images (Pneumonia)** 数据集，使用 PyTorch + ResNet-50 构建二分类模型，并配以 PyQt5 桌面应用实现单张 X 光图像的肺炎检测。同时引入 Rust 预处理工具，将 JPEG 图像批量转换为 `.npy` 缓存以加速训练。

## 项目结构

```text
/root/image_detector
├── src-py/                 # Python 核心代码
│   ├── config.py           # 路径与超参数配置
│   ├── dataset.py          # 数据集加载、分层划分、DataLoader
│   ├── model.py            # ResNet-50 二分类模型
│   ├── train.py            # 训练脚本
│   ├── evaluate.py         # 评估与可视化
│   ├── inference.py        # 单图推理
│   ├── gui.py              # PyQt5 桌面应用
│   ├── image_process.py    # 图像预处理辅助函数
│   ├── download_database.py# 数据集下载（KaggleHub）
│   └── main.py             # CLI 入口
├── src/bin/preprocess.rs   # Rust 图像预处理缓存工具
├── models/                 # 保存训练好的模型（best_model.pth）
├── cache/                  # Rust 生成的 .npy 缓存
├── outputs/                # 评估图表与 metrics.json
├── pyproject.toml          # uv 依赖配置
└── Cargo.toml              # Rust 依赖配置
```

## 环境要求

- Python 3.12
- uv（Python 包管理）
- Rust + Cargo（可选，用于生成图像缓存）
- NVIDIA GPU（推荐，已测试 RTX 3080 Ti + CUDA 12.x）

## 安装依赖

```bash
cd /root/image_detector
uv sync
```

主要依赖：PyTorch、torchvision、PyQt5、matplotlib、seaborn、scikit-learn、numpy、pandas、tqdm。

## 数据准备

数据集默认路径：

```text
/root/autodl-tmp/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/chest_xray
```

若数据不存在，可运行：

```bash
uv run python src-py/download_database.py
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

- `cache/train_images.npy` / `cache/train_labels.npy`
- `cache/test_images.npy` / `cache/test_labels.npy`

### 2. 训练模型

```bash
uv run python src-py/main.py train
```

训练完成后，最佳模型保存至 `models/best_model.pth`，训练历史保存至 `outputs/history.json`。

### 3. 评估模型

```bash
uv run python src-py/main.py evaluate
```

评估输出：

- `outputs/metrics.json`：验证集与测试集指标
- `outputs/roc_curve.png`：ROC 曲线
- `outputs/confusion_matrix.png`：混淆矩阵
- `outputs/training_history.png`：训练历史曲线

### 4. 启动 GUI

```bash
uv run python src-py/main.py gui
```

在界面中选择胸部 X 光图像，点击「开始检测」，即可查看预测结果与置信度。

## 模型设计

- 基础网络：`torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`
- 输出层：替换 `fc` 为 `nn.Linear(2048, 1)`，输出单 logit
- 损失函数：`BCEWithLogitsLoss(pos_weight)`，使用 `normal_count / pneumonia_count` 作为正例权重，缓解类别不平衡
- 优化器：AdamW，学习率 1e-4，weight decay 1e-4
- 学习率调度：ReduceLROnPlateau（监控验证 F1）
- 早停： patience=7

## 数据划分

官方验证集仅 16 张，本项目合并官方 `train` + `val`（共 5216 张），使用 `sklearn.model_selection.train_test_split` 按 85%/15% 分层抽样得到新的训练集与验证集，官方 `test` 集作为最终测试集。

## 性能指标

在测试集上评估：Accuracy、Precision、Recall、F1-Score、AUC-ROC，并绘制 ROC 曲线与混淆矩阵。

