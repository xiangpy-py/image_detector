# 基于胸部 X 光图像的肺炎检测系统 — 实施计划

## 背景与目标

本项目要求基于 Kaggle 的 Chest X-Ray Images (Pneumonia) 数据集，构建一个胸部 X 光肺炎检测系统。核心能力包括：数据预处理与增强、基于 ResNet-50 的二分类模型训练、完整的模型评估（Accuracy / Precision / Recall / F1 / AUC / ROC / 混淆矩阵）、以及基于 PyQt5 的桌面 GUI 推理界面。用户额外要求：使用 uv 管理依赖；必要时使用 Rust；并保留一份记录写作过程的文本。

当前项目状态：

- Python 3.12 + uv 0.11.25 已就绪，`.venv` 已创建。
- `pyproject.toml` 仅含 `kagglehub`、`opencv-python`、`scikit-learn`。
- `src-py/` 下现有代码为占位/不完整：`main.py` 为 hello world，`image_process.py` 语法不完整，`download_database.py` 可用。
- Rust 项目为 `src/main.rs` 的 hello world + 已配置清华镜像的 `Cargo.toml`。
- 数据集已存在：`/root/autodl-tmp/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/chest_xray`，含 train/val/test；官方 val 集仅 16 张，需要重新划分。
- GPU 为 NVIDIA RTX 3080 Ti，因此 PyTorch 按 CUDA 12.1 安装。

## 推荐方案

采用 **Python 主流程 + Rust 图像预处理缓存** 的混合架构：

- Python：负责数据加载、模型训练、评估、可视化、GUI 和推理。
- Rust：负责将 JPEG 图像批量解码、缩放、归一化并保存为 `.npy` 缓存，供 Python 训练时直接读取，减少训练前预处理耗时。
- 额外产出：一份 `WRITING_LOG.md`，记录从环境搭建到各模块实现的过程、关键决策与踩坑点。

## 项目结构

```text
/root/image_detector
├── pyproject.toml                  # uv 依赖（加入 torch、PyQt5、matplotlib 等）
├── uv.lock
├── Cargo.toml                      # Rust 项目配置（扩展预处理 bin）
├── src/
│   └── bin/preprocess.rs           # Rust 图像预处理缓存工具
├── src-py/
│   ├── config.py                   # 路径、超参数、类别映射
│   ├── dataset.py                  # 数据集划分、DataLoader、缓存读取
│   ├── model.py                    # ResNet-50 二分类模型
│   ├── train.py                    # 训练循环、早停、checkpoint
│   ├── evaluate.py                 # 评估指标、ROC、混淆矩阵、训练曲线
│   ├── inference.py                # 单图推理接口
│   ├── gui.py                      # PyQt5 桌面应用
│   ├── image_process.py            # 图像预处理辅助函数（重写）
│   ├── download_database.py        # 数据集下载（保留）
│   └── main.py                     # CLI 入口：train / eval / gui
├── models/                         # 保存 best_model.pth（加入 .gitignore）
├── outputs/                        # ROC、混淆矩阵、训练历史、metrics.json
├── cache/                          # Rust 生成的 .npy 缓存（加入 .gitignore）
├── README.md                       # 项目说明与运行方式
└── WRITING_LOG.md                  # 写作/开发过程记录
```

## 关键实施步骤

### 1. 环境搭建（uv）

在 `/root/image_detector` 下执行：

```bash
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv add PyQt5 matplotlib seaborn pillow pandas numpy tqdm
```

说明：

- 检测到 GPU 为 RTX 3080 Ti，因此使用 CUDA 12.1 的 PyTorch wheel。
- 不额外引入 `albumentations` / `tensorboard`，避免增加复杂度；数据增强使用 `torchvision.transforms`。

### 2. Rust 图像预处理缓存

扩展 `Cargo.toml` 增加 `clap`（可选，命令行参数）与已有 `image`、`ndarray`、`ndarray-npy`、`walkdir`。

实现 `src/bin/preprocess.rs`：

- 读取 `DATASET_ROOT` 下所有 `NORMAL` / `PNEUMONIA` 的 JPEG。
- 将图像解码并缩放到 224×224，按 ImageNet 均值/标准差归一化。
- 输出为 `cache/train_images.npy`、`cache/train_labels.npy`、`cache/test_images.npy`、`cache/test_labels.npy`。
- Python `dataset.py` 优先读取缓存；缓存不存在时回退到 `torchvision.datasets.ImageFolder`。

运行方式：

```bash
cargo run --release --bin preprocess -- --root <数据集路径> --out cache/
```

### 3. 数据准备与划分

数据集实际根目录：

```python
DATASET_ROOT = Path("/root/autodl-tmp/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/chest_xray")
```

处理策略：

- 合并官方 `train` + `val`（共 5216 张）。
- 使用 `sklearn.model_selection.train_test_split` 做分层抽样（stratify），85% 训练 / 15% 验证。
- 官方 `test` 作为最终测试集不动。

预处理与增强：

- 训练：`Resize(256) → RandomCrop(224) → RandomHorizontalFlip → RandomRotation(10°) → ColorJitter → ToTensor → Normalize(ImageNet)`。
- 验证/测试/GUI：`Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet)`。

### 4. 模型设计

`src-py/model.py`：

- 使用 `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`。
- 替换 `model.fc` 为 `nn.Linear(2048, 1)`。
- 输出 logit，配合 `BCEWithLogitsLoss`。

### 5. 训练策略

`src-py/train.py`：

- 类别不平衡处理：使用 `pos_weight = torch.tensor([normal_count / pneumonia_count])`，约为 0.346，通过 `BCEWithLogitsLoss(pos_weight=...)` 增大肺炎类别权重。
- 优化器：`AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)`。
- 学习率调度：`ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)`，监控验证 F1。
- 早停：patience=7，监控验证 F1。
- Checkpoint：保存验证 F1 最高的模型到 `models/best_model.pth`。
- 默认超参数：epochs=20，batch_size=32，input_size=224。

### 6. 模型评估

`src-py/evaluate.py`：

- 在验证集和测试集上分别计算：Accuracy、Precision、Recall、F1-Score、AUC。
- 绘制并保存：
  - `outputs/roc_curve.png`
  - `outputs/confusion_matrix.png`
  - `outputs/training_history.png`
- 输出 `outputs/metrics.json`。

### 7. GUI 开发

`src-py/gui.py` + `src-py/inference.py`：

- PyQt5 单窗口应用。
- 按钮：选择图像、开始检测、退出。
- 左侧为图像预览，右侧显示预测类别（NORMAL / PNEUMONIA）与置信度进度条。
- 加载 `models/best_model.pth`，使用 `val_transform` 预处理并推理。
- 异常处理：模型缺失、读取失败、非图像文件等弹出 `QMessageBox`。

### 8. CLI 入口

`src-py/main.py` 支持：

```bash
uv run python src-py/main.py train          # 训练
uv run python src-py/main.py evaluate       # 在 test 集评估并绘图
uv run python src-py/main.py gui            # 启动 GUI
uv run python src-py/main.py cache          # 调用 Rust 生成缓存
```

### 9. 文档

- `README.md`：项目简介、环境安装、运行命令、文件说明。
- `WRITING_LOG.md`：记录开发过程，包括环境配置、数据划分决策、Rust 缓存实现、训练调参、GUI 开发、遇到的问题与解决方案。

## 待修改/新增的关键文件

- `/root/image_detector/pyproject.toml`
- `/root/image_detector/Cargo.toml`
- `/root/image_detector/src/bin/preprocess.rs`
- `/root/image_detector/src-py/config.py`
- `/root/image_detector/src-py/dataset.py`
- `/root/image_detector/src-py/model.py`
- `/root/image_detector/src-py/train.py`
- `/root/image_detector/src-py/evaluate.py`
- `/root/image_detector/src-py/inference.py`
- `/root/image_detector/src-py/gui.py`
- `/root/image_detector/src-py/image_process.py`
- `/root/image_detector/src-py/main.py`
- `/root/image_detector/README.md`
- `/root/image_detector/WRITING_LOG.md`
- `/root/image_detector/.gitignore`（加入 models/、cache/、outputs/、__pycache__ 等）

## 验证方法

1. 依赖验证：`uv run python -c "import torch; print(torch.cuda.is_available())"` 应输出 `True`。
2. Rust 缓存验证：运行 `cargo run --release --bin preprocess` 后，`cache/` 下生成 `.npy` 文件，形状分别为 `(N, 3, 224, 224)` 和 `(N,)`。
3. 数据划分验证：合并后总样本 5216，训练约 4433，验证约 783，分层后两类比例与原始一致。
4. 训练验证：loss 下降，验证 AUC > 0.85（预期 0.95 左右），`models/best_model.pth` 成功保存。
5. 评估验证：`outputs/` 下生成 ROC 曲线、混淆矩阵、训练历史图和 `metrics.json`。
6. GUI 验证：选择 test 集图像后，返回 NORMAL/PNEUMONIA 及置信度，与 `evaluate.py` 结果一致。

## 预期产出

- 可训练的 ResNet-50 二分类模型。
- 完整的评估报告与可视化图表。
- 可交互的 PyQt5 桌面应用。
- Rust 实现的图像预处理缓存工具。
- 项目说明 README 与开发过程记录 WRITING_LOG。
