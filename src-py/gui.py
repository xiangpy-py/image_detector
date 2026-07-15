"""胸部 X 光肺炎检测系统 — 图形界面 (GUI)。

替代命令行，提供训练、缓存生成、模型加载、图像检测、评估等一站式操作界面。
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QTabWidget,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QCheckBox,
    QGridLayout,
    QDialog,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CACHE_DIR,
    CACHE_SIZE,
    CLASS_NAMES,
    DATASET_ROOT,
    DATASET_ROOTS,
    DEFAULT_THRESHOLD,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    LABEL_MAP,
    LEARNING_RATE,
    MODELS_DIR,
    NUM_WORKERS,
    OUTPUTS_DIR,
    PIN_MEMORY,
    RANDOM_SEED,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    VAL_SIZE,
    WEIGHT_DECAY,
    override_paths,
)
from dataset import CachedDataset, get_class_counts, load_cached_data
from image_process import preprocess_image_path
from inference import load_trained_model, predict
from metrics import evaluate_model, get_loss_function, get_pos_weight
from model import build_model, set_seed
from threshold_tuner import find_best_threshold, load_threshold, save_threshold

# ───────────────────────────────────────────────────────────
#  工具函数
# ───────────────────────────────────────────────────────────


def _style_button(btn, color="#3B82F6", text_color="white"):
    """统一按钮样式——现代圆角风格。"""
    btn.setStyleSheet(
        f"""
        QPushButton {{
            background-color: {color};
            color: {text_color};
            border: none;
            padding: 8px 18px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 13px;
        }}
        QPushButton:hover {{
            background-color: {color};
            opacity: 0.85;
        }}
        QPushButton:pressed {{
            background-color: {color};
            opacity: 0.7;
        }}
        QPushButton:disabled {{
            background-color: #E2E8F0;
            color: #94A3B8;
        }}
    """
    )


def _style_group(title):
    """统一分组框样式——现代卡片风格。"""
    g = QGroupBox(title)
    g.setStyleSheet(
        """
        QGroupBox {
            font-weight: bold;
            font-size: 14px;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            margin-top: 12px;
            padding-top: 16px;
            padding-left: 14px;
            padding-right: 14px;
            padding-bottom: 14px;
            background: #FFFFFF;
            color: #1E293B;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 14px;
            padding: 0 8px;
            color: #475569;
            font-size: 13px;
        }
    """
    )
    return g


def _clear_layout(layout):
    """递归清空布局中的所有子部件。"""
    if layout is None:
        return
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        else:
            _clear_layout(item.layout())


class NumericTableItem(QTableWidgetItem):
    """支持按数值排序的表格项。"""

    def __init__(self, value, text=None):
        super().__init__(text if text is not None else f"{value:.4f}")
        self.setData(Qt.UserRole, float(value))

    def __lt__(self, other):
        val = self.data(Qt.UserRole)
        other_val = other.data(Qt.UserRole)
        if val is None:
            return other_val is not None
        if other_val is None:
            return False
        return val < other_val


# ───────────────────────────────────────────────────────────
#  Worker 线程
# ───────────────────────────────────────────────────────────


class TrainWorker(QThread):
    """后台训练线程，实时发射日志和指标信号。"""

    log_signal = Signal(str)
    epoch_signal = Signal(int, dict)  # epoch, metrics_dict
    progress_signal = Signal(int, int)  # current, total
    finished_signal = Signal(bool, str)

    def __init__(self, epochs, lr, batch_size, weight_decay, patience, resume_from=None, parent=None):
        super().__init__(parent)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.patience = patience
        self.resume_from = resume_from

    def _log(self, msg: str):
        self.log_signal.emit(msg)

    def run(self):
        try:
            set_seed()
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._log(f"🖥️ 使用设备: {device}")

            # 加载数据
            self._log("📂 正在加载数据集...")
            train_images, train_labels = load_cached_data("train")
            test_images, test_labels = load_cached_data("test")
            self._log(f"训练集类别分布: {get_class_counts(train_labels)}")

            from sklearn.model_selection import train_test_split

            train_idx, val_idx = train_test_split(
                np.arange(len(train_labels)),
                test_size=VAL_SIZE,
                stratify=train_labels,
                random_state=RANDOM_SEED,
            )

            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(IMG_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
            val_transform = transforms.Compose(
                [
                    transforms.CenterCrop(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )

            train_dataset = CachedDataset(train_images[train_idx], train_labels[train_idx], transform=train_transform)
            val_dataset = CachedDataset(train_images[val_idx], train_labels[val_idx], transform=val_transform)
            test_dataset = CachedDataset(test_images, test_labels, transform=val_transform)

            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                persistent_workers=NUM_WORKERS > 0,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                persistent_workers=NUM_WORKERS > 0,
            )

            # 构建模型
            model = build_model(pretrained=True).to(device)
            pos_weight = get_pos_weight(train_labels, device)
            criterion = get_loss_function(device, pos_weight=pos_weight)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
            )

            use_amp = device.type == "cuda"
            scaler = GradScaler(device=str(device)) if use_amp else None
            amp_enabled = use_amp

            start_epoch = 0
            best_f1 = 0.0
            patience_counter = 0
            history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}

            # 恢复训练
            if self.resume_from:
                resume_path = Path(self.resume_from)
                if resume_path.exists():
                    self._log(f"🔄 从 checkpoint 恢复训练: {resume_path}")
                    checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    if "scheduler_state_dict" in checkpoint:
                        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    start_epoch = checkpoint.get("epoch", 0) + 1
                    best_f1 = checkpoint.get("best_f1", 0.0)
                    self._log(f"恢复至 epoch {start_epoch}, 当前最佳 val_f1={best_f1:.4f}")
                else:
                    self._log(f"⚠️ checkpoint 不存在: {resume_path}")

            self._log(f"🏋️ 开始训练: epochs={self.epochs}, lr={self.lr}, batch_size={self.batch_size}")

            for epoch in range(start_epoch, self.epochs):
                self.progress_signal.emit(epoch + 1, self.epochs)
                model.train()
                running_loss = 0.0
                pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
                for images, labels in pbar:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
                    optimizer.zero_grad()
                    with autocast(device_type=device.type, enabled=amp_enabled):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    if amp_enabled:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item() * images.size(0)
                    pbar.set_postfix({"loss": loss.item()})

                train_loss = running_loss / len(train_loader.dataset)
                val_metrics, _, _ = evaluate_model(model, val_loader, device, criterion)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_metrics["loss"])
                history["val_f1"].append(val_metrics["f1"])
                history["val_auc"].append(val_metrics["auc"])

                log_msg = (
                    f"Epoch {epoch + 1}/{self.epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"val_acc={val_metrics['accuracy']:.4f} | "
                    f"val_f1={val_metrics['f1']:.4f} | "
                    f"val_auc={val_metrics['auc']:.4f}"
                )
                self._log(log_msg)
                self.epoch_signal.emit(epoch + 1, {
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_f1": val_metrics["f1"],
                    "val_auc": val_metrics["auc"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                })

                # 实时保存 history，供图表更新使用
                history_path = OUTPUTS_DIR / "history.json"
                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)

                scheduler.step(val_metrics["f1"])

                if val_metrics["f1"] > best_f1:
                    best_f1 = val_metrics["f1"]
                    patience_counter = 0
                    best_path = MODELS_DIR / "best_model.pth"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "best_f1": best_f1,
                        },
                        best_path,
                    )
                    self._log(f"⭐ 最佳模型已保存，val_f1={best_f1:.4f} -> {best_path}")
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    self._log(f"🛑 早停触发 ( patience={self.patience} )")
                    break

            history_path = OUTPUTS_DIR / "history.json"
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            self._log(f"📊 训练历史已保存至 {history_path}")
            self.finished_signal.emit(True, f"训练完成！最佳 val_f1={best_f1:.4f}")

        except Exception as e:
            err = traceback.format_exc()
            self._log(f"❌ 训练失败: {e}\n{err}")
            self.finished_signal.emit(False, str(e))


class EvaluateWorker(QThread):
    """后台评估线程。"""

    log_signal = Signal(str)
    result_signal = Signal(dict)
    finished_signal = Signal(bool, str)

    def _log(self, msg: str):
        self.log_signal.emit(msg)

    def run(self):
        try:
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._log(f"🖥️ 使用设备: {device}")

            self._log("📂 正在加载数据...")
            train_images, train_labels = load_cached_data("train")
            test_images, test_labels = load_cached_data("test")

            val_transform = transforms.Compose(
                [
                    transforms.CenterCrop(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )

            from sklearn.model_selection import train_test_split

            train_idx, val_idx = train_test_split(
                np.arange(len(train_labels)),
                test_size=VAL_SIZE,
                stratify=train_labels,
                random_state=RANDOM_SEED,
            )
            val_dataset = CachedDataset(train_images[val_idx], train_labels[val_idx], transform=val_transform)
            test_dataset = CachedDataset(test_images, test_labels, transform=val_transform)

            val_loader = DataLoader(
                val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                persistent_workers=NUM_WORKERS > 0,
            )
            test_loader = DataLoader(
                test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                persistent_workers=NUM_WORKERS > 0,
            )

            model_path = MODELS_DIR / "best_model.pth"
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            model = build_model(pretrained=False)
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            self._log(f"✅ 模型已加载: {model_path}")

            pos_weight = get_pos_weight(train_labels, device)
            criterion = get_loss_function(device, pos_weight=pos_weight)

            # 验证集
            self._log("📊 正在评估验证集...")
            val_metrics, val_labels, val_probs = evaluate_model(model, val_loader, device, criterion)
            best_threshold, best_f1 = find_best_threshold(val_labels, val_probs, metric="f1")
            threshold_path = save_threshold(best_threshold)
            self._log(f"✅ 验证集最优阈值: {best_threshold:.4f} (F1={best_f1:.4f})，已保存")

            # 测试集
            self._log("📊 正在评估测试集...")
            test_preds = (np.array(val_probs) >= best_threshold).astype(int)
            test_metrics, test_labels, test_probs = evaluate_model(model, test_loader, device, criterion)
            test_preds = (np.array(test_probs) >= best_threshold).astype(int)
            test_metrics["threshold"] = best_threshold

            # 保存图表
            self._log("📈 正在生成图表...")

            # ROC
            fpr, tpr, _ = roc_curve(test_labels, test_probs)
            auc = roc_auc_score(test_labels, test_probs)
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(OUTPUTS_DIR / "roc_curve.png", dpi=150)

            # Confusion Matrix
            cm = confusion_matrix(test_labels, test_preds)
            fig2 = Figure(figsize=(6, 5))
            ax2 = fig2.add_subplot(111)
            im = ax2.imshow(cm, interpolation="nearest", cmap="Blues")
            ax2.set_title("Confusion Matrix")
            fig2.colorbar(im, ax=ax2)
            tick_marks = np.arange(len(CLASS_NAMES))
            ax2.set_xticks(tick_marks)
            ax2.set_yticks(tick_marks)
            ax2.set_xticklabels(CLASS_NAMES)
            ax2.set_yticklabels(CLASS_NAMES)
            ax2.set_ylabel("True")
            ax2.set_xlabel("Predicted")
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax2.text(
                        j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14,
                    )
            fig2.tight_layout()
            fig2.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=150)

            # Training History
            history_path = OUTPUTS_DIR / "history.json"
            if history_path.exists():
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                epochs = range(1, len(history["train_loss"]) + 1)
                fig3 = Figure(figsize=(12, 4))
                ax3 = fig3.add_subplot(1, 3, 1)
                ax3.plot(epochs, history["train_loss"], label="Train Loss")
                ax3.plot(epochs, history["val_loss"], label="Val Loss")
                ax3.set_xlabel("Epoch")
                ax3.set_ylabel("Loss")
                ax3.legend()
                ax3.set_title("Loss Curve")

                ax4 = fig3.add_subplot(1, 3, 2)
                ax4.plot(epochs, history["val_f1"], label="Val F1")
                ax4.set_xlabel("Epoch")
                ax4.set_ylabel("F1 Score")
                ax4.legend()
                ax4.set_title("Validation F1")

                ax5 = fig3.add_subplot(1, 3, 3)
                ax5.plot(epochs, history["val_auc"], label="Val AUC")
                ax5.set_xlabel("Epoch")
                ax5.set_ylabel("AUC")
                ax5.legend()
                ax5.set_title("Validation AUC")
                fig3.tight_layout()
                fig3.savefig(OUTPUTS_DIR / "training_history.png", dpi=150)
                self._log("📈 训练历史图已保存")

            # 保存报告
            report = {
                "val": val_metrics,
                "test": test_metrics,
                "threshold": best_threshold,
            }
            report_path = OUTPUTS_DIR / "metrics.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            self._log(f"📄 评估报告已保存至 {report_path}")

            self.result_signal.emit(report)
            self.finished_signal.emit(True, "评估完成！")

        except Exception as e:
            err = traceback.format_exc()
            self._log(f"❌ 评估失败: {e}\n{err}")
            self.finished_signal.emit(False, str(e))


class PythonCacheWorker(QThread):
    """纯 Python 生成图像缓存（Rust 不可用时 fallback）。"""

    log_signal = Signal(str)
    progress_signal = Signal(int, int)
    finished_signal = Signal(bool, str)

    def __init__(self, dataset_roots, cache_dir, cache_size, parent=None):
        super().__init__(parent)
        # 支持单个 Path 或列表
        if isinstance(dataset_roots, (str, Path)):
            self.dataset_roots = [Path(dataset_roots)]
        else:
            self.dataset_roots = [Path(r) for r in dataset_roots]
        self.cache_dir = Path(cache_dir)
        self.cache_size = cache_size

    def _log(self, msg: str):
        self.log_signal.emit(msg)

    def _process_one_root(self, root: Path, out_dir: Path, is_multi: bool):
        """处理单个数据集根目录，返回 (image_count, [image_arrays], [labels])。"""
        self._log(f"📂 数据集根目录: {root}")
        self._log(f"💾 输出目录: {out_dir}")
        self._log(f"📐 图像尺寸: {self.cache_size}")

        transform = transforms.Compose(
            [
                transforms.Resize(self.cache_size),
                transforms.CenterCrop(self.cache_size),
            ]
        )

        total = 0
        for split in ["train", "test"]:
            split_dir = root / split
            if not split_dir.exists():
                continue
            for cls_name in CLASS_NAMES:
                cls_dir = split_dir / cls_name
                if cls_dir.exists():
                    total += len(list(cls_dir.glob("*.jpeg")))
                    total += len(list(cls_dir.glob("*.jpg")))
                    total += len(list(cls_dir.glob("*.png")))

        self._log(f"🗂️  预计处理图像总数: {total}")
        processed = 0
        total_saved = 0

        for split in ["train", "test"]:
            split_dir = root / split
            if not split_dir.exists():
                self._log(f"⚠️ 目录不存在，跳过: {split_dir}")
                continue

            images = []
            labels = []
            for cls_name in CLASS_NAMES:
                cls_dir = split_dir / cls_name
                if not cls_dir.exists():
                    continue
                label = LABEL_MAP[cls_name]
                files = []
                for ext in ["*.jpeg", "*.jpg", "*.png"]:
                    files.extend(cls_dir.glob(ext))
                for f in sorted(files):
                    try:
                        img = Image.open(f).convert("RGB")
                        img = transform(img)
                        img_arr = np.array(img)
                        img_arr = np.transpose(img_arr, (2, 0, 1))
                        images.append(img_arr)
                        labels.append(label)
                        processed += 1
                        if processed % 100 == 0:
                            self.progress_signal.emit(processed, total)
                            self._log(f"  已处理 {processed}/{total} ...")
                    except Exception as e:
                        self._log(f"⚠️ 跳过 {f}: {e}")

            if images:
                images_arr = np.array(images, dtype=np.uint8)
                labels_arr = np.array(labels, dtype=np.int64)
                np.save(out_dir / f"{split}_images.npy", images_arr)
                np.save(out_dir / f"{split}_labels.npy", labels_arr)
                self._log(f"✅ {split} 缓存已保存: {len(images)} 张图像")
                total_saved += len(images)
            else:
                self._log(f"⚠️ {split} 无图像数据")

        self.progress_signal.emit(processed, total)
        return total_saved

    def run(self):
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            is_multi = len(self.dataset_roots) > 1

            total_processed = 0
            merged_info = {"datasets": []}

            for i, root in enumerate(self.dataset_roots):
                if not root.exists():
                    self._log(f"⚠️ 数据集根目录不存在，跳过: {root}")
                    continue

                base_name = root.name
                subdir_name = base_name
                counter = 1
                while (self.cache_dir / subdir_name).exists():
                    subdir_name = f"{base_name}_{counter}"
                    counter += 1

                out_dir = self.cache_dir if not is_multi else self.cache_dir / subdir_name
                out_dir.mkdir(parents=True, exist_ok=True)

                if is_multi:
                    self._log(f"[{i + 1}/{len(self.dataset_roots)}] 处理: {root} -> {subdir_name}")

                n = self._process_one_root(root, out_dir, is_multi)
                total_processed += n

                if is_multi:
                    merged_info["datasets"].append({
                        "name": subdir_name,
                        "root": str(root),
                        "cache_subdir": str(out_dir),
                    })

            if is_multi and merged_info["datasets"]:
                import json
                info_path = self.cache_dir / "merged_info.json"
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(merged_info, f, indent=2, ensure_ascii=False)
                self._log(f"📋 合并信息已保存: merged_info.json")

            self.finished_signal.emit(True, f"缓存生成完成！共处理 {total_processed} 张图像")

        except Exception as e:
            err = traceback.format_exc()
            self._log(f"❌ 缓存生成失败: {e}\n{err}")
            self.finished_signal.emit(False, str(e))


class RustCacheWorker(QThread):
    """尝试使用 Rust 预处理，失败时自动 fallback 到 Python。"""

    log_signal = Signal(str)
    progress_signal = Signal(int, int)
    finished_signal = Signal(bool, str)

    def __init__(self, dataset_roots, cache_dir, cache_size, parent=None):
        super().__init__(parent)
        # 支持单个 Path 或列表
        if isinstance(dataset_roots, (str, Path)):
            self.dataset_roots = [Path(dataset_roots)]
        else:
            self.dataset_roots = [Path(r) for r in dataset_roots]
        self.cache_dir = Path(cache_dir)
        self.cache_size = cache_size
        self._python_worker = None

    def _log(self, msg: str):
        self.log_signal.emit(msg)

    def run(self):
        try:
            from rust_preprocessor import preprocess_dataset

            is_multi = len(self.dataset_roots) > 1
            self._log("🦀 使用 Rust 加速预处理...")

            total_train = 0
            total_test = 0
            merged_info = {"datasets": []}

            for i, root in enumerate(self.dataset_roots):
                if not root.exists():
                    self._log(f"⚠️ 数据集根目录不存在，跳过: {root}")
                    continue

                # 子缓存目录名
                base_name = root.name
                subdir_name = base_name
                counter = 1
                while (self.cache_dir / subdir_name).exists():
                    subdir_name = f"{base_name}_{counter}"
                    counter += 1

                out_dir = self.cache_dir if not is_multi else self.cache_dir / subdir_name
                out_dir.mkdir(parents=True, exist_ok=True)

                if is_multi:
                    self._log(f"[{i + 1}/{len(self.dataset_roots)}] 处理: {root} -> {subdir_name}")
                else:
                    self._log(f"📂 数据集根目录: {root}")

                train_count, test_count = preprocess_dataset(
                    str(root), str(out_dir), self.cache_size
                )
                total_train += train_count
                total_test += test_count

                if is_multi:
                    merged_info["datasets"].append({
                        "name": subdir_name,
                        "root": str(root),
                        "cache_subdir": str(out_dir),
                    })

            if is_multi and merged_info["datasets"]:
                import json
                info_path = self.cache_dir / "merged_info.json"
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(merged_info, f, indent=2, ensure_ascii=False)
                self._log(f"📋 合并信息已保存: merged_info.json")

            self._log(f"✅ Rust 预处理完成: 训练集 {total_train} 张, 测试集 {total_test} 张")
            self.finished_signal.emit(
                True, f"Rust 预处理完成: 训练集 {total_train} 张, 测试集 {total_test} 张"
            )
        except ImportError:
            self._log("⚠️ Rust 扩展未安装，将使用 Python 模式生成缓存...")
            self._python_worker = PythonCacheWorker(
                self.dataset_roots, self.cache_dir, self.cache_size
            )
            self._python_worker.log_signal.connect(self.log_signal)
            self._python_worker.progress_signal.connect(self.progress_signal)
            self._python_worker.finished_signal.connect(self.finished_signal)
            self._python_worker.run()
        except Exception as e:
            err = traceback.format_exc()
            self._log(f"❌ Rust 预处理失败: {e}\n{err}")
            self._log("🔄 尝试使用 Python 模式...")
            self._python_worker = PythonCacheWorker(
                self.dataset_roots, self.cache_dir, self.cache_size
            )
            self._python_worker.log_signal.connect(self.log_signal)
            self._python_worker.progress_signal.connect(self.progress_signal)
            self._python_worker.finished_signal.connect(self.finished_signal)
            self._python_worker.run()


class DownloadWorker(QThread):
    """后台下载数据集。"""

    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def _log(self, msg: str):
        self.log_signal.emit(msg)

    def run(self):
        try:
            import os
            from system import get_default_kaggle_cache_dir
            cache_dir = get_default_kaggle_cache_dir()
            os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)
            self._log("⬇️ 开始下载 Kaggle 数据集...")
            import kagglehub
            path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
            self._log(f"✅ 下载完成: {path}")
            self._log(f"📂 内容: {os.listdir(path)}")
            self.finished_signal.emit(True, f"数据集已下载至: {path}")
        except Exception as e:
            err = traceback.format_exc()
            self._log(f"❌ 下载失败: {e}\n{err}")
            self.finished_signal.emit(False, str(e))


class InferenceWorker(QThread):
    """推理线程。"""

    result_ready = Signal(str, float, float)
    error_occurred = Signal(str)

    def __init__(self, image_path, model, device, threshold=DEFAULT_THRESHOLD, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.model = model
        self.device = device
        self.threshold = threshold

    def run(self):
        try:
            class_name, confidence, prob = predict(
                self.image_path, self.model, self.device, threshold=self.threshold
            )
            self.result_ready.emit(class_name, confidence, prob)
        except Exception as e:
            self.error_occurred.emit(str(e))


class BatchInferenceWorker(QThread):
    """批量推理线程。"""

    progress_signal = Signal(int, int)
    result_ready = Signal(list)
    error_occurred = Signal(str)
    log_signal = Signal(str)

    def __init__(self, image_paths, model, device, threshold=DEFAULT_THRESHOLD, batch_size=32, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.model = model
        self.device = device
        self.threshold = threshold
        self.batch_size = batch_size

    def run(self):
        try:
            from inference import _infer_batch
            results = []
            total = len(self.image_paths)
            for i in range(0, total, self.batch_size):
                batch_paths = self.image_paths[i:i + self.batch_size]
                probs = _infer_batch(batch_paths, self.model, self.device)
                for path, prob in zip(batch_paths, probs):
                    label = 1 if prob >= self.threshold else 0
                    class_name = CLASS_NAMES[label]
                    confidence = prob if label == 1 else 1 - prob
                    results.append((path, class_name, float(confidence), float(prob)))
                self.progress_signal.emit(min(i + self.batch_size, total), total)
            self.result_ready.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))


# ───────────────────────────────────────────────────────────
#  主窗口
# ───────────────────────────────────────────────────────────


class PneumoniaGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("胸部 X 光肺炎检测系统")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # 状态变量
        self.model = None
        self.device = None
        self.threshold = DEFAULT_THRESHOLD
        self.image_path = None
        self._original_pixmap = None

        self.train_worker = None
        self.evaluate_worker = None
        self.cache_worker = None
        self.download_worker = None
        self.inference_worker = None
        self.batch_worker = None

        self._build_ui()
        self._load_model()
        self._check_dataset()

    def _build_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 标题栏 - 带背景色
        title = QLabel("🫁 胸部 X 光肺炎检测系统")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #FFFFFF; "
            "padding: 14px; background-color: #2563EB; border-radius: 10px;"
        )
        main_layout.addWidget(title)

        # 标签页
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                background: #FFFFFF;
                padding: 8px;
            }
            QTabBar::tab {
                background: #F1F5F9;
                border: 1px solid #E2E8F0;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 10px 28px;
                font-weight: bold;
                font-size: 13px;
                color: #64748B;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #FFFFFF;
                color: #2563EB;
                border-bottom: 2px solid #2563EB;
            }
            QTabBar::tab:hover:!selected {
                background: #E2E8F0;
                color: #475569;
            }
        """
        )

        self.tabs.addTab(self._build_inference_tab(), "🔍 图像检测")
        self.tabs.addTab(self._build_training_tab(), "🏋️ 模型训练")
        self.tabs.addTab(self._build_evaluation_tab(), "📊 模型评估")
        self.tabs.addTab(self._build_dataset_tab(), "🗂️ 数据集管理")

        main_layout.addWidget(self.tabs, 1)

        # 状态栏
        status_layout = QHBoxLayout()
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet(
            "color: #64748B; font-size: 12px; padding: 6px 12px; "
            "background: #F1F5F9; border-radius: 6px;"
        )
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()

        # 模型状态
        self.model_status_label = QLabel("模型: 未加载")
        self.model_status_label.setStyleSheet(
            "color: #EF4444; font-size: 12px; padding: 6px 12px; "
            "background: #FEF2F2; border-radius: 6px; font-weight: bold;"
        )
        status_layout.addWidget(self.model_status_label)
        status_layout.addSpacing(16)

        # 数据集状态
        self.dataset_status_label = QLabel(f"数据集: {DATASET_ROOT}")
        self.dataset_status_label.setStyleSheet(
            "color: #64748B; font-size: 12px; padding: 6px 12px; "
            "background: #F1F5F9; border-radius: 6px;"
        )
        self.dataset_status_label.setToolTip(str(DATASET_ROOT))
        status_layout.addWidget(self.dataset_status_label)

        main_layout.addLayout(status_layout)
        self.setLayout(main_layout)

        # 全局样式
        self.setStyleSheet("""
            QWidget {
                font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
                background-color: #F8FAFC;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 8px 10px;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                background: #FFFFFF;
                color: #1E293B;
                font-size: 13px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #3B82F6;
            }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 20px;
                border: none;
                background: #F1F5F9;
                border-radius: 4px;
            }
            QTextEdit {
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                background: #0F172A;
                color: #E2E8F0;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 12px;
                padding: 8px;
            }
            QTableWidget {
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                background: #FFFFFF;
                gridline-color: #F1F5F9;
                alternate-background-color: #F8FAFC;
                selection-background-color: #DBEAFE;
                selection-color: #1E293B;
            }
            QTableWidget::item {
                padding: 6px;
                border-bottom: 1px solid #F1F5F9;
            }
            QHeaderView::section {
                background-color: #F1F5F9;
                color: #475569;
                padding: 10px 12px;
                font-weight: bold;
                font-size: 12px;
                border: none;
                border-bottom: 2px solid #E2E8F0;
            }
            QProgressBar {
                border: none;
                border-radius: 8px;
                text-align: center;
                height: 22px;
                background: #E2E8F0;
                font-weight: bold;
                font-size: 11px;
                color: #475569;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3B82F6, stop:1 #60A5FA);
                border-radius: 8px;
            }
            QGroupBox {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 16px;
                padding-left: 14px;
                padding-right: 14px;
                padding-bottom: 14px;
                font-weight: bold;
                font-size: 14px;
                color: #1E293B;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 8px;
                color: #475569;
                font-size: 13px;
            }
            QCheckBox {
                font-size: 13px;
                color: #334155;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #CBD5E1;
                background: #FFFFFF;
            }
            QCheckBox::indicator:checked {
                background: #3B82F6;
                border: 1px solid #3B82F6;
            }
            QPushButton {
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                padding: 8px 18px;
            }
            QPushButton:hover {
                opacity: 0.9;
            }
            QPushButton:disabled {
                background-color: #E2E8F0;
                color: #94A3B8;
                border: none;
            }
            QDialog {
                background: #FFFFFF;
                border-radius: 12px;
            }
            QMessageBox {
                background: #FFFFFF;
            }
        """)

    # ───────────────────────────────────────────────
    #  检测标签页
    # ───────────────────────────────────────────────

    def _build_inference_tab(self):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # 左侧: 图像 + 控制
        left = QVBoxLayout()
        left.setSpacing(12)

        # 模型选择
        model_group = _style_group("模型")
        model_layout = QHBoxLayout()
        self.model_path_label = QLabel("自动加载最佳模型")
        self.model_path_label.setStyleSheet("color: #64748B; font-size: 12px;")
        model_layout.addWidget(self.model_path_label, 1)
        self.load_model_btn = QPushButton("📂 选择模型")
        self.load_model_btn.setToolTip("手动选择 .pth 模型文件")
        self.load_model_btn.clicked.connect(self.select_model_file)
        _style_button(self.load_model_btn, "#64748B")
        model_layout.addWidget(self.load_model_btn)
        self.reload_model_btn = QPushButton("🔄 重新加载")
        self.reload_model_btn.clicked.connect(self._load_model)
        _style_button(self.reload_model_btn, "#64748B")
        model_layout.addWidget(self.reload_model_btn)
        model_group.setLayout(model_layout)
        left.addWidget(model_group)

        # 图像显示
        img_group = _style_group("图像预览")
        img_layout = QVBoxLayout()
        self.image_label = QLabel("请选择一张胸部 X 光图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "border: 2px dashed #CBD5E1; background-color: #F8FAFC; color: #94A3B8; "
            "font-size: 16px; border-radius: 10px;"
        )
        self.image_label.setMinimumSize(480, 480)
        img_layout.addWidget(self.image_label)
        img_group.setLayout(img_layout)
        left.addWidget(img_group, 1)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        self.select_btn = QPushButton("📂 选择图像")
        self.select_btn.clicked.connect(self.select_image)
        _style_button(self.select_btn, "#3B82F6")
        btn_layout.addWidget(self.select_btn)

        self.detect_btn = QPushButton("🔍 开始检测")
        self.detect_btn.clicked.connect(self.detect)
        self.detect_btn.setEnabled(False)
        _style_button(self.detect_btn, "#10B981")
        btn_layout.addWidget(self.detect_btn)

        self.batch_btn = QPushButton("📁 批量检测")
        self.batch_btn.clicked.connect(self.batch_detect)
        _style_button(self.batch_btn, "#F59E0B")
        btn_layout.addWidget(self.batch_btn)

        left.addLayout(btn_layout)
        layout.addLayout(left, 3)

        # 右侧: 结果
        right = QVBoxLayout()
        right.setSpacing(12)

        result_group = _style_group("检测结果")
        result_layout = QVBoxLayout()

        self.result_title = QLabel("尚未检测")
        self.result_title.setAlignment(Qt.AlignCenter)
        self.result_title.setStyleSheet(
            "font-size: 28px; font-weight: bold; color: #94A3B8; padding: 16px;"
        )
        result_layout.addWidget(self.result_title)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setFormat("置信度: %p%")
        result_layout.addWidget(self.confidence_bar)

        self.result_detail = QTextEdit()
        self.result_detail.setReadOnly(True)
        self.result_detail.setPlaceholderText("检测详情将显示在这里...")
        self.result_detail.setMaximumHeight(120)
        result_layout.addWidget(self.result_detail)

        result_group.setLayout(result_layout)
        right.addWidget(result_group)

        # 批量结果表格
        batch_group = _style_group("批量检测结果")
        batch_layout = QVBoxLayout()
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(5)
        self.batch_table.setHorizontalHeaderLabels(["文件名", "预测结果", "置信度", "肺炎概率", "预览"])
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        self.batch_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.batch_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.batch_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.batch_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.batch_table.setSortingEnabled(True)
        self.batch_table.setMaximumHeight(250)
        batch_layout.addWidget(self.batch_table)
        batch_group.setLayout(batch_layout)
        right.addWidget(batch_group)

        layout.addLayout(right, 2)
        widget.setLayout(layout)
        return widget

    # ───────────────────────────────────────────────
    #  训练标签页
    # ───────────────────────────────────────────────

    def _build_training_tab(self):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # 左侧: 参数 + 控制 + 日志
        left = QVBoxLayout()
        left.setSpacing(12)

        # 参数设置
        param_group = _style_group("训练参数")
        param_form = QFormLayout()

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(EPOCHS)
        param_form.addRow("Epochs:", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(1e-5)
        self.lr_spin.setValue(LEARNING_RATE)
        param_form.addRow("Learning Rate:", self.lr_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(BATCH_SIZE)
        param_form.addRow("Batch Size:", self.batch_spin)

        self.wd_spin = QDoubleSpinBox()
        self.wd_spin.setRange(0, 1.0)
        self.wd_spin.setDecimals(6)
        self.wd_spin.setValue(WEIGHT_DECAY)
        param_form.addRow("Weight Decay:", self.wd_spin)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 50)
        self.patience_spin.setValue(EARLY_STOP_PATIENCE)
        param_form.addRow("Early Stop Patience:", self.patience_spin)

        self.resume_check = QCheckBox("从 checkpoint 恢复")
        self.resume_check.setChecked(False)
        param_form.addRow(self.resume_check)

        self.resume_path_edit = QLineEdit()
        self.resume_path_edit.setPlaceholderText("模型路径 (可选)")
        self.resume_path_edit.setEnabled(False)
        self.resume_check.toggled.connect(self.resume_path_edit.setEnabled)
        self.resume_browse_btn = QPushButton("浏览...")
        self.resume_browse_btn.clicked.connect(self._browse_resume_path)
        self.resume_browse_btn.setEnabled(False)
        self.resume_check.toggled.connect(self.resume_browse_btn.setEnabled)
        resume_layout = QHBoxLayout()
        resume_layout.addWidget(self.resume_path_edit, 1)
        resume_layout.addWidget(self.resume_browse_btn)
        param_form.addRow("Checkpoint:", resume_layout)

        param_group.setLayout(param_form)
        left.addWidget(param_group)

        # 控制按钮
        train_btn_layout = QHBoxLayout()
        self.start_train_btn = QPushButton("▶️ 开始训练")
        self.start_train_btn.clicked.connect(self.start_training)
        _style_button(self.start_train_btn, "#4CAF50")
        train_btn_layout.addWidget(self.start_train_btn)

        self.stop_train_btn = QPushButton("⏹️ 停止训练")
        self.stop_train_btn.clicked.connect(self.stop_training)
        self.stop_train_btn.setEnabled(False)
        _style_button(self.stop_train_btn, "#F44336")
        train_btn_layout.addWidget(self.stop_train_btn)
        left.addLayout(train_btn_layout)

        # 训练进度
        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(0)
        self.train_progress.setFormat("Epoch %v / %m")
        left.addWidget(self.train_progress)

        # 日志
        log_group = _style_group("训练日志")
        log_layout = QVBoxLayout()
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setPlaceholderText("训练日志将显示在这里...")
        log_layout.addWidget(self.train_log)
        log_group.setLayout(log_layout)
        left.addWidget(log_group, 1)

        layout.addLayout(left, 1)

        # 右侧: 实时图表
        right = QVBoxLayout()
        right.setSpacing(12)

        chart_group = _style_group("实时训练曲线")
        chart_layout = QVBoxLayout()

        self.train_figure = Figure(figsize=(8, 6), facecolor="white")
        self.train_canvas = FigureCanvas(self.train_figure)
        chart_layout.addWidget(self.train_canvas)

        chart_group.setLayout(chart_layout)
        right.addWidget(chart_group, 1)

        # 当前指标
        metric_group = _style_group("当前指标")
        metric_layout = QGridLayout()
        metric_layout.setSpacing(10)

        self.metric_labels = {}
        metrics = [
            ("Train Loss", "train_loss"),
            ("Val Loss", "val_loss"),
            ("Val F1", "val_f1"),
            ("Val AUC", "val_auc"),
            ("Val Acc", "val_accuracy"),
            ("Val Precision", "val_precision"),
            ("Val Recall", "val_recall"),
        ]
        for i, (name, key) in enumerate(metrics):
            lbl = QLabel(f"{name}: -")
            lbl.setStyleSheet("font-size: 13px; color: #424242;")
            lbl.setAlignment(Qt.AlignCenter)
            metric_layout.addWidget(lbl, i // 4, i % 4)
            self.metric_labels[key] = lbl

        metric_group.setLayout(metric_layout)
        right.addWidget(metric_group)

        layout.addLayout(right, 1)
        widget.setLayout(layout)
        return widget

    # ───────────────────────────────────────────────
    #  评估标签页
    # ───────────────────────────────────────────────

    def _build_evaluation_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # 顶部控制
        top_layout = QHBoxLayout()
        self.run_eval_btn = QPushButton("▶️ 运行评估")
        self.run_eval_btn.clicked.connect(self.run_evaluation)
        _style_button(self.run_eval_btn, "#2196F3")
        top_layout.addWidget(self.run_eval_btn)

        self.eval_progress = QProgressBar()
        self.eval_progress.setRange(0, 0)
        self.eval_progress.setValue(0)
        self.eval_progress.setVisible(False)
        top_layout.addWidget(self.eval_progress, 1)
        layout.addLayout(top_layout)

        # 指标表格
        metric_group = _style_group("评估指标")
        metric_layout = QVBoxLayout()
        self.eval_table = QTableWidget()
        self.eval_table.setColumnCount(3)
        self.eval_table.setHorizontalHeaderLabels(["指标", "验证集", "测试集"])
        self.eval_table.horizontalHeader().setStretchLastSection(True)
        self.eval_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.eval_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.eval_table.setMaximumHeight(200)
        metric_layout.addWidget(self.eval_table)
        metric_group.setLayout(metric_layout)
        layout.addWidget(metric_group)

        # 日志
        eval_log_group = _style_group("评估日志")
        eval_log_layout = QVBoxLayout()
        self.eval_log = QTextEdit()
        self.eval_log.setReadOnly(True)
        self.eval_log.setPlaceholderText("评估日志将显示在这里...")
        eval_log_layout.addWidget(self.eval_log)
        eval_log_group.setLayout(eval_log_layout)
        layout.addWidget(eval_log_group, 1)

        # 图表按钮
        chart_btn_layout = QHBoxLayout()
        self.view_roc_btn = QPushButton("📈 查看 ROC 曲线")
        self.view_roc_btn.clicked.connect(lambda: self._view_image(OUTPUTS_DIR / "roc_curve.png"))
        _style_button(self.view_roc_btn, "#9C27B0")
        chart_btn_layout.addWidget(self.view_roc_btn)

        self.view_cm_btn = QPushButton("📊 查看混淆矩阵")
        self.view_cm_btn.clicked.connect(lambda: self._view_image(OUTPUTS_DIR / "confusion_matrix.png"))
        _style_button(self.view_cm_btn, "#9C27B0")
        chart_btn_layout.addWidget(self.view_cm_btn)

        self.view_hist_btn = QPushButton("📉 查看训练历史")
        self.view_hist_btn.clicked.connect(lambda: self._view_image(OUTPUTS_DIR / "training_history.png"))
        _style_button(self.view_hist_btn, "#9C27B0")
        chart_btn_layout.addWidget(self.view_hist_btn)
        layout.addLayout(chart_btn_layout)

        widget.setLayout(layout)
        return widget

    # ───────────────────────────────────────────────
    #  数据集标签页
    # ───────────────────────────────────────────────

    def _build_dataset_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # 当前数据集
        current_group = _style_group("当前数据集")
        current_layout = QHBoxLayout()
        self.current_dataset_label = QLabel(f"📂 {DATASET_ROOT}")
        self.current_dataset_label.setStyleSheet("font-size: 13px; color: #333;")
        self.current_dataset_label.setWordWrap(True)
        current_layout.addWidget(self.current_dataset_label, 1)
        self.change_dataset_btn = QPushButton("📂 更改数据集")
        self.change_dataset_btn.clicked.connect(self.select_dataset)
        _style_button(self.change_dataset_btn, "#2196F3")
        current_layout.addWidget(self.change_dataset_btn)
        current_group.setLayout(current_layout)
        layout.addWidget(current_group)

        # 缓存生成
        cache_group = _style_group("图像缓存")
        cache_layout = QHBoxLayout()
        self.cache_info_label = QLabel(f"💾 缓存目录: {CACHE_DIR}")
        self.cache_info_label.setStyleSheet("font-size: 13px; color: #333;")
        cache_layout.addWidget(self.cache_info_label, 1)
        self.gen_cache_btn = QPushButton("⚡ 生成缓存")
        self.gen_cache_btn.clicked.connect(self.run_cache_generation)
        _style_button(self.gen_cache_btn, "#FF9800")
        cache_layout.addWidget(self.gen_cache_btn)
        cache_group.setLayout(cache_layout)
        layout.addWidget(cache_group)

        # 下载数据集
        download_group = _style_group("数据集下载")
        download_layout = QHBoxLayout()
        self.download_info = QLabel("⬇️ 从 Kaggle 下载 chest-xray-pneumonia 数据集")
        self.download_info.setStyleSheet("font-size: 13px; color: #333;")
        download_layout.addWidget(self.download_info, 1)
        self.download_btn = QPushButton("⬇️ 下载数据集")
        self.download_btn.clicked.connect(self.run_download_dataset)
        _style_button(self.download_btn, "#4CAF50")
        download_layout.addWidget(self.download_btn)
        download_group.setLayout(download_layout)
        layout.addWidget(download_group)

        # 缓存生成日志
        cache_log_group = _style_group("处理日志")
        cache_log_layout = QVBoxLayout()
        self.dataset_log = QTextEdit()
        self.dataset_log.setReadOnly(True)
        self.dataset_log.setPlaceholderText("缓存生成和下载日志将显示在这里...")
        cache_log_layout.addWidget(self.dataset_log)
        cache_log_group.setLayout(cache_log_layout)
        layout.addWidget(cache_log_group, 1)

        # 缓存进度
        self.cache_progress = QProgressBar()
        self.cache_progress.setRange(0, 100)
        self.cache_progress.setValue(0)
        layout.addWidget(self.cache_progress)

        widget.setLayout(layout)
        return widget

    # ───────────────────────────────────────────────
    #  通用方法
    # ───────────────────────────────────────────────

    def _load_model(self):
        try:
            self.model, self.device = load_trained_model()
            self.threshold = load_threshold()
            self.model_status_label.setText(
                f"模型: 已加载 (threshold={self.threshold:.4f})"
            )
            self.model_status_label.setStyleSheet("color: #15803D; font-size: 12px; padding: 4px; background: #F0FDF4; border-radius: 4px; font-weight: bold;")
            self.model_path_label.setText(f"已加载: {MODELS_DIR / 'best_model.pth'}")
        except Exception as e:
            self.model = None
            self.device = None
            self.model_status_label.setText("模型: 未加载")
            self.model_status_label.setStyleSheet("color: #EF4444; font-size: 12px; padding: 4px; background: #FEF2F2; border-radius: 4px; font-weight: bold;")
            self.model_path_label.setText(f"加载失败: {e}")

    def _check_dataset(self):
        train_images = CACHE_DIR / "train_images.npy"
        train_labels = CACHE_DIR / "train_labels.npy"
        if not train_images.exists() or not train_labels.exists():
            self.status_label.setText("⚠️ 缓存文件不存在，请先生成缓存")
            self.status_label.setStyleSheet("color: #B45309; font-size: 12px; padding: 4px; background: #FFFBEB; border-radius: 4px; font-weight: bold;")
        else:
            self.status_label.setText("✅ 就绪")
            self.status_label.setStyleSheet("color: #15803D; font-size: 12px; padding: 4px; background: #F0FDF4; border-radius: 4px; font-weight: bold;")

    # ───────────────────────────────────────────────
    #  检测页面方法
    # ───────────────────────────────────────────────

    def select_model_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", str(MODELS_DIR), "Model Files (*.pth)"
        )
        if not path:
            return
        try:
            self.model, self.device = load_trained_model(model_path=path)
            self.threshold = load_threshold()
            self.model_path_label.setText(f"已加载: {path}")
            self.model_status_label.setText(
                f"模型: 已加载 (threshold={self.threshold:.4f})"
            )
            self.model_status_label.setStyleSheet("color: #15803D; font-size: 12px; padding: 4px; background: #F0FDF4; border-radius: 4px; font-weight: bold;")
            QMessageBox.information(self, "成功", "模型加载成功！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {e}")

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择胸部 X 光图像", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not path:
            return
        self.image_path = path
        self._original_pixmap = QPixmap(path)
        self._update_image_display()
        self.detect_btn.setEnabled(True)
        self.result_title.setText("尚未检测")
        self.result_title.setStyleSheet("font-size: 28px; font-weight: bold; color: #94A3B8; padding: 16px;")
        self.confidence_bar.setValue(0)
        self.result_detail.setPlainText(f"已选择图像: {path}\n点击「开始检测」进行推理。")

    def _update_image_display(self):
        if self._original_pixmap and not self._original_pixmap.isNull():
            scaled = self._original_pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
            self.image_label.setStyleSheet("border: 2px solid #CBD5E1; background-color: #F8FAFC;")

    def detect(self):
        if self.image_path is None:
            QMessageBox.warning(self, "提示", "请先选择图像。")
            return
        if self.model is None:
            QMessageBox.warning(self, "提示", "模型未加载，请先加载模型。")
            return

        self.detect_btn.setEnabled(False)
        self.result_detail.setPlainText("正在检测，请稍候...")

        if self.inference_worker is not None:
            self.inference_worker.deleteLater()

        self.inference_worker = InferenceWorker(
            self.image_path, self.model, self.device, threshold=self.threshold
        )
        self.inference_worker.result_ready.connect(self._on_result_ready)
        self.inference_worker.error_occurred.connect(self._on_inference_error)
        self.inference_worker.finished.connect(lambda: self.detect_btn.setEnabled(True))
        self.inference_worker.start()

    def batch_detect(self):
        if self.model is None:
            QMessageBox.warning(self, "提示", "模型未加载，请先加载模型。")
            return
        directory = QFileDialog.getExistingDirectory(self, "选择包含图像的文件夹")
        if not directory:
            return
        dir_path = Path(directory)
        files = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            files.extend(dir_path.glob(ext))
        if not files:
            QMessageBox.information(self, "提示", "未找到图像文件。")
            return

        reply = QMessageBox.question(
            self, "确认", f"找到 {len(files)} 张图像，是否开始批量检测？"
        )
        if reply != QMessageBox.Yes:
            return

        self.batch_btn.setEnabled(False)
        self.result_detail.setPlainText(f"批量检测中: 0/{len(files)} ...")

        if self.batch_worker is not None:
            self.batch_worker.deleteLater()

        self.batch_worker = BatchInferenceWorker(
            [str(f) for f in files], self.model, self.device, threshold=self.threshold
        )
        self.batch_worker.progress_signal.connect(self._on_batch_progress)
        self.batch_worker.result_ready.connect(self._on_batch_result)
        self.batch_worker.error_occurred.connect(self._on_inference_error)
        self.batch_worker.finished.connect(lambda: self.batch_btn.setEnabled(True))
        self.batch_worker.start()

    def _on_batch_progress(self, current, total):
        self.result_detail.setPlainText(f"批量检测中: {current}/{total} ...")

    def _on_batch_result(self, results):
        self.batch_table.setSortingEnabled(False)
        self.batch_table.setRowCount(len(results))
        for i, (path, class_name, confidence, prob) in enumerate(results):
            self.batch_table.setItem(i, 0, QTableWidgetItem(Path(path).name))
            self.batch_table.setItem(i, 1, QTableWidgetItem(class_name))
            self.batch_table.setItem(i, 2, NumericTableItem(confidence))
            self.batch_table.setItem(i, 3, NumericTableItem(prob))
            # 预览按钮
            preview_btn = QPushButton("👁️ 预览")
            preview_btn.setCursor(Qt.PointingHandCursor)
            preview_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #607D8B;
                    color: white;
                    border: none;
                    padding: 4px 10px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #455A64;
                }
            """
            )
            preview_btn.clicked.connect(lambda checked=False, p=path: self._preview_image(p))
            self.batch_table.setCellWidget(i, 4, preview_btn)
        self.batch_table.setSortingEnabled(True)
        self.result_detail.setPlainText(f"批量检测完成，共 {len(results)} 张图像。")

    def _preview_image(self, path: str):
        """弹窗预览指定图像。"""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"预览 - {Path(path).name}")
        dialog.resize(900, 900)
        layout = QVBoxLayout()
        lbl = QLabel()
        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            lbl.setPixmap(pixmap.scaled(880, 880, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            lbl.setText("无法加载图像")
            lbl.setAlignment(Qt.AlignCenter)
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
        dialog.setLayout(layout)
        dialog.exec()

    def _on_result_ready(self, class_name, confidence, prob):
        self.result_title.setText(class_name)
        if class_name == "PNEUMONIA":
            self.result_title.setStyleSheet(
                "font-size: 28px; font-weight: bold; color: #EF4444; background: #FEF2F2; padding: 16px; border-radius: 10px;"
            )
        else:
            self.result_title.setStyleSheet(
                "font-size: 28px; font-weight: bold; color: #10B981; background: #F0FDF4; padding: 16px; border-radius: 10px;"
            )
        self.confidence_bar.setValue(int(confidence * 100))
        self.result_detail.setPlainText(
            f"图像路径: {self.image_path}\n"
            f"预测类别: {class_name}\n"
            f"置信度: {confidence:.4f}\n"
            f"肺炎概率: {prob:.4f}\n"
            f"正常概率: {1 - prob:.4f}\n"
            f"检测阈值: {self.threshold:.4f}"
        )

    def _on_inference_error(self, message):
        QMessageBox.critical(self, "检测失败", f"推理过程中出现错误: {message}")
        self.result_detail.setPlainText(f"检测失败: {message}")

    # ───────────────────────────────────────────────
    #  训练页面方法
    # ───────────────────────────────────────────────

    def _browse_resume_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 Checkpoint", str(MODELS_DIR), "Model Files (*.pth)"
        )
        if path:
            self.resume_path_edit.setText(path)

    def start_training(self):
        epochs = self.epochs_spin.value()
        lr = self.lr_spin.value()
        batch_size = self.batch_spin.value()
        weight_decay = self.wd_spin.value()
        patience = self.patience_spin.value()
        resume_from = None
        if self.resume_check.isChecked() and self.resume_path_edit.text().strip():
            resume_from = self.resume_path_edit.text().strip()

        reply = QMessageBox.question(
            self, "确认", f"开始训练?\nEpochs={epochs}, LR={lr}, Batch={batch_size}"
        )
        if reply != QMessageBox.Yes:
            return

        self.train_log.clear()
        self.start_train_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)
        self.train_progress.setValue(0)
        self.train_progress.setRange(0, epochs)

        # 清空图表
        self.train_figure.clear()
        self.train_canvas.draw()

        # 清空指标
        for lbl in self.metric_labels.values():
            lbl.setText("-")

        if self.train_worker is not None:
            self.train_worker.deleteLater()

        self.train_worker = TrainWorker(
            epochs, lr, batch_size, weight_decay, patience, resume_from=resume_from
        )
        self.train_worker.log_signal.connect(self._on_train_log)
        self.train_worker.epoch_signal.connect(self._on_train_epoch)
        self.train_worker.progress_signal.connect(self.train_progress.setValue)
        self.train_worker.finished_signal.connect(self._on_train_finished)
        self.train_worker.start()

    def stop_training(self):
        if self.train_worker and self.train_worker.isRunning():
            self.train_worker.terminate()
            self.train_worker.wait(1000)
            self.train_log.append("⏹️ 训练已停止")
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)

    def _on_train_log(self, msg: str):
        self.train_log.append(msg)
        # 自动滚动到底部
        scrollbar = self.train_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_train_epoch(self, epoch: int, metrics: dict):
        # 更新指标
        for key, lbl in self.metric_labels.items():
            val = metrics.get(key)
            if val is not None:
                lbl.setText(f"{val:.4f}")

        # 更新图表
        self._update_train_plot()

    def _update_train_plot(self):
        history_path = OUTPUTS_DIR / "history.json"
        if not history_path.exists():
            return
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            return

        epochs = range(1, len(history["train_loss"]) + 1)
        self.train_figure.clear()

        ax1 = self.train_figure.add_subplot(2, 2, 1)
        ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
        ax1.plot(epochs, history["val_loss"], "r-s", label="Val Loss", markersize=3)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.set_title("Loss Curve")
        ax1.grid(True, alpha=0.3)

        ax2 = self.train_figure.add_subplot(2, 2, 2)
        ax2.plot(epochs, history["val_f1"], "g-^", label="Val F1", markersize=3)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("F1 Score")
        ax2.legend()
        ax2.set_title("Validation F1")
        ax2.grid(True, alpha=0.3)

        ax3 = self.train_figure.add_subplot(2, 2, 3)
        ax3.plot(epochs, history["val_auc"], "m-d", label="Val AUC", markersize=3)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("AUC")
        ax3.legend()
        ax3.set_title("Validation AUC")
        ax3.grid(True, alpha=0.3)

        ax4 = self.train_figure.add_subplot(2, 2, 4)
        if len(epochs) > 1:
            ax4.text(0.5, 0.5, f"Epochs: {len(epochs)}\nBest F1: {max(history['val_f1']):.4f}",
                     ha="center", va="center", fontsize=14, transform=ax4.transAxes)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis("off")
        ax4.set_title("Summary")

        self.train_figure.tight_layout()
        self.train_canvas.draw()

    def _on_train_finished(self, success: bool, msg: str):
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        if success:
            QMessageBox.information(self, "训练完成", msg)
            self._load_model()
        else:
            QMessageBox.critical(self, "训练失败", msg)

    # ───────────────────────────────────────────────
    #  评估页面方法
    # ───────────────────────────────────────────────

    def run_evaluation(self):
        reply = QMessageBox.question(self, "确认", "开始运行模型评估？")
        if reply != QMessageBox.Yes:
            return

        self.eval_log.clear()
        self.run_eval_btn.setEnabled(False)
        self.eval_progress.setVisible(True)

        if self.evaluate_worker is not None:
            self.evaluate_worker.deleteLater()

        self.evaluate_worker = EvaluateWorker()
        self.evaluate_worker.log_signal.connect(self.eval_log.append)
        self.evaluate_worker.result_signal.connect(self._on_eval_result)
        self.evaluate_worker.finished_signal.connect(self._on_eval_finished)
        self.evaluate_worker.start()

    def _on_eval_result(self, report: dict):
        self.eval_table.setRowCount(0)
        metrics_map = {
            "Accuracy": "accuracy",
            "Precision": "precision",
            "Recall": "recall",
            "F1 Score": "f1",
            "AUC": "auc",
            "Loss": "loss",
        }
        row = 0
        for name, key in metrics_map.items():
            val_val = report.get("val", {}).get(key)
            test_val = report.get("test", {}).get(key)
            if val_val is not None or test_val is not None:
                self.eval_table.insertRow(row)
                self.eval_table.setItem(row, 0, QTableWidgetItem(name))
                self.eval_table.setItem(row, 1, QTableWidgetItem(f"{val_val:.4f}" if val_val is not None else "-"))
                self.eval_table.setItem(row, 2, QTableWidgetItem(f"{test_val:.4f}" if test_val is not None else "-"))
                row += 1
        # Threshold
        threshold = report.get("threshold")
        if threshold is not None:
            self.eval_table.insertRow(row)
            self.eval_table.setItem(row, 0, QTableWidgetItem("Optimal Threshold"))
            self.eval_table.setItem(row, 1, QTableWidgetItem("-"))
            self.eval_table.setItem(row, 2, QTableWidgetItem(f"{threshold:.4f}"))

    def _on_eval_finished(self, success: bool, msg: str):
        self.run_eval_btn.setEnabled(True)
        self.eval_progress.setVisible(False)
        if success:
            QMessageBox.information(self, "评估完成", msg)
            self._load_model()
        else:
            QMessageBox.critical(self, "评估失败", msg)

    def _view_image(self, path: Path):
        if not path.exists():
            QMessageBox.information(self, "提示", f"文件尚未生成: {path}")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(path.name)
        dialog.resize(800, 600)
        layout = QVBoxLayout()
        lbl = QLabel()
        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            lbl.setPixmap(pixmap.scaled(780, 580, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
        dialog.setLayout(layout)
        dialog.exec()

    # ───────────────────────────────────────────────
    #  数据集页面方法
    # ───────────────────────────────────────────────

    def select_dataset(self):
        path = QFileDialog.getExistingDirectory(
            self, "选择数据集根目录（需包含 train/val/test 子目录）", str(DATASET_ROOT),
        )
        if not path:
            return
        override_paths(dataset_root=path)
        self.current_dataset_label.setText(f"📂 {DATASET_ROOT}")
        self.dataset_status_label.setText(f"数据集: {DATASET_ROOT}")
        self.dataset_status_label.setToolTip(str(DATASET_ROOT))
        self.dataset_log.append(f"✅ 数据集路径已更新: {path}")
        self._check_dataset()

    def run_cache_generation(self):
        num_datasets = len(DATASET_ROOTS)
        if num_datasets == 1:
            msg = f"生成图像缓存?\n数据集: {DATASET_ROOT}\n缓存目录: {CACHE_DIR}"
        else:
            msg = (
                f"生成图像缓存?\n"
                f"数据集数量: {num_datasets}\n"
                f"缓存目录: {CACHE_DIR}\n\n"
                f"将处理以下数据集:\n" +
                "\n".join(f"  • {r}" for r in DATASET_ROOTS)
            )
        reply = QMessageBox.question(self, "确认", msg)
        if reply != QMessageBox.Yes:
            return

        self.gen_cache_btn.setEnabled(False)
        self.dataset_log.clear()
        self.cache_progress.setValue(0)

        if self.cache_worker is not None:
            self.cache_worker.deleteLater()

        self.cache_worker = RustCacheWorker(DATASET_ROOTS, CACHE_DIR, CACHE_SIZE)
        self.cache_worker.log_signal.connect(self.dataset_log.append)
        self.cache_worker.progress_signal.connect(self._on_cache_progress)
        self.cache_worker.finished_signal.connect(self._on_cache_finished)
        self.cache_worker.start()

    def _on_cache_progress(self, current, total):
        if total > 0:
            self.cache_progress.setRange(0, total)
            self.cache_progress.setValue(current)

    def _on_cache_finished(self, success, msg):
        self.gen_cache_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "缓存完成", msg)
            self._check_dataset()
        else:
            QMessageBox.critical(self, "缓存失败", msg)

    def run_download_dataset(self):
        reply = QMessageBox.question(self, "确认", "开始从 Kaggle 下载数据集？")
        if reply != QMessageBox.Yes:
            return

        self.download_btn.setEnabled(False)
        self.dataset_log.clear()

        if self.download_worker is not None:
            self.download_worker.deleteLater()

        self.download_worker = DownloadWorker()
        self.download_worker.log_signal.connect(self.dataset_log.append)
        self.download_worker.finished_signal.connect(self._on_download_finished)
        self.download_worker.start()

    def _on_download_finished(self, success, msg):
        self.download_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "下载完成", msg)
        else:
            QMessageBox.critical(self, "下载失败", msg)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_image_display()


# ───────────────────────────────────────────────────────────
#  入口
# ───────────────────────────────────────────────────────────


def main():
    app = QApplication(sys.argv)
    window = PneumoniaGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
