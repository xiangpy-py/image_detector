import sys

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
)

from config import CACHE_DIR, CACHE_SIZE, DATASET_ROOT, DEFAULT_THRESHOLD, override_paths
from inference import load_trained_model, predict
from threshold_tuner import load_threshold


class InferenceWorker(QThread):
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


class PreprocessWorker(QThread):
    """在后台线程运行 Rust 预处理，避免阻塞 GUI。"""

    finished = Signal(bool, str)

    def __init__(self, dataset_root, cache_dir, cache_size, parent=None):
        super().__init__(parent)
        self.dataset_root = dataset_root
        self.cache_dir = cache_dir
        self.cache_size = cache_size

    def run(self):
        try:
            from rust_preprocessor import preprocess_dataset

            train_count, test_count = preprocess_dataset(
                str(self.dataset_root),
                str(self.cache_dir),
                self.cache_size,
            )
            self.finished.emit(
                True, f"预处理完成: 训练集 {train_count} 张, 测试集 {test_count} 张"
            )
        except Exception as e:
            self.finished.emit(False, str(e))


class PneumoniaGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("胸部 X 光肺炎检测系统")
        self.setMinimumSize(1000, 600)

        self.model = None
        self.device = None
        self.image_path = None
        self.worker = None
        self.preprocess_worker = None
        self.threshold = DEFAULT_THRESHOLD
        self._original_pixmap = None

        self._build_ui()
        self._load_model()
        self._check_dataset()

    def _build_ui(self):
        main_layout = QHBoxLayout()

        # ─── 左侧：图像 + 检测 ───
        left_layout = QVBoxLayout()

        # 数据集路径显示
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("数据集:"))
        self.dataset_label = QLabel(str(DATASET_ROOT))
        self.dataset_label.setStyleSheet("color: gray; font-size: 11px;")
        self.dataset_label.setWordWrap(True)
        dataset_layout.addWidget(self.dataset_label, 1)
        self.dataset_btn = QPushButton("更改")
        self.dataset_btn.setToolTip("选择数据集根目录（需包含 train/val/test 子目录）")
        self.dataset_btn.clicked.connect(self.select_dataset)
        dataset_layout.addWidget(self.dataset_btn)
        self.preprocess_btn = QPushButton("预处理")
        self.preprocess_btn.setToolTip("调用 Rust 模块生成图像缓存")
        self.preprocess_btn.clicked.connect(self.run_preprocess)
        dataset_layout.addWidget(self.preprocess_btn)
        left_layout.addLayout(dataset_layout)

        self.image_label = QLabel("请选择一张胸部 X 光图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.image_label.setMinimumSize(400, 400)
        left_layout.addWidget(self.image_label)

        button_layout = QHBoxLayout()
        self.select_btn = QPushButton("选择图像")
        self.select_btn.clicked.connect(self.select_image)
        self.detect_btn = QPushButton("开始检测")
        self.detect_btn.clicked.connect(self.detect)
        self.detect_btn.setEnabled(False)
        self.quit_btn = QPushButton("退出")
        self.quit_btn.clicked.connect(self.close)
        button_layout.addWidget(self.select_btn)
        button_layout.addWidget(self.detect_btn)
        button_layout.addWidget(self.quit_btn)
        left_layout.addLayout(button_layout)

        # ─── 右侧：结果 ───
        right_layout = QVBoxLayout()

        self.result_title = QLabel("检测结果")
        self.result_title.setAlignment(Qt.AlignCenter)
        self.result_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        right_layout.addWidget(self.result_title)

        self.result_label = QLabel("尚未检测")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 24px; color: gray;")
        right_layout.addWidget(self.result_label)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(True)
        right_layout.addWidget(self.confidence_bar)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setPlaceholderText("检测详情将显示在这里...")
        right_layout.addWidget(self.detail_text)

        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)
        self.setLayout(main_layout)

    def _check_dataset(self):
        """检查数据集和缓存是否存在，给出友好提示。"""
        train_images = CACHE_DIR / "train_images.npy"
        train_labels = CACHE_DIR / "train_labels.npy"
        if not train_images.exists() or not train_labels.exists():
            self.detail_text.setPlainText(
                "⚠ 缓存文件不存在。\n"
                "1. 点击「更改」选择数据集根目录\n"
                "2. 点击「预处理」生成图像缓存\n"
                "3. 然后在 CLI 中运行: uv run python src-py/main.py train"
            )

    def _load_model(self):
        try:
            self.model, self.device = load_trained_model()
            self.threshold = load_threshold()
            self.detail_text.setPlainText(
                f"模型加载成功。\n"
                f"当前检测阈值: {self.threshold:.4f} "
                f"({'已加载优化阈值' if self.threshold != DEFAULT_THRESHOLD else '使用默认阈值 0.5'})"
            )
        except Exception as e:
            self.detail_text.setPlainText(
                f"⚠ 模型未加载: {e}\n"
                f"请先运行训练: uv run python src-py/main.py train"
            )

    def select_dataset(self):
        """打开文件夹对话框选择数据集根目录。"""
        path = QFileDialog.getExistingDirectory(
            self,
            "选择数据集根目录（需包含 train/val/test 子目录）",
            str(DATASET_ROOT),
        )
        if not path:
            return

        override_paths(dataset_root=path)
        self.dataset_label.setText(str(DATASET_ROOT))
        self.detail_text.setPlainText(f"数据集路径已更新: {path}")
        self._check_dataset()

    def run_preprocess(self):
        """在后台线程调用 Rust 预处理模块。"""
        self.preprocess_btn.setEnabled(False)
        self.detail_text.setPlainText(
            f"正在预处理数据...\n"
            f"数据集: {DATASET_ROOT}\n"
            f"缓存目录: {CACHE_DIR}"
        )

        if self.preprocess_worker is not None:
            self.preprocess_worker.deleteLater()

        self.preprocess_worker = PreprocessWorker(
            DATASET_ROOT, CACHE_DIR, CACHE_SIZE
        )
        self.preprocess_worker.finished.connect(self._on_preprocess_finished)
        self.preprocess_worker.start()

    def _on_preprocess_finished(self, success, message):
        self.preprocess_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "预处理完成", message)
            self.detail_text.setPlainText(message)
        else:
            QMessageBox.critical(self, "预处理失败", message)
            self.detail_text.setPlainText(f"预处理失败: {message}")

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
        self.result_label.setText("尚未检测")
        self.result_label.setStyleSheet("font-size: 24px; color: gray;")
        self.confidence_bar.setValue(0)
        self.detail_text.setPlainText(f"已选择图像：{path}")

    def _update_image_display(self):
        if self._original_pixmap and not self._original_pixmap.isNull():
            scaled = self._original_pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)

    def detect(self):
        if self.image_path is None or self.model is None:
            QMessageBox.warning(self, "提示", "请先选择图像并确保模型已加载。")
            return

        self.detect_btn.setEnabled(False)
        self.detail_text.setPlainText(
            f"正在检测，请稍候...\n当前阈值: {self.threshold:.4f}"
        )

        if self.worker is not None:
            self.worker.deleteLater()

        self.worker = InferenceWorker(
            self.image_path, self.model, self.device, threshold=self.threshold
        )
        self.worker.result_ready.connect(self._on_result_ready)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished.connect(lambda: self.detect_btn.setEnabled(True))
        self.worker.start()

    def _on_result_ready(self, class_name, confidence, prob):
        self.result_label.setText(class_name)
        if class_name == "PNEUMONIA":
            self.result_label.setStyleSheet(
                "font-size: 24px; color: red; font-weight: bold;"
            )
        else:
            self.result_label.setStyleSheet(
                "font-size: 24px; color: green; font-weight: bold;"
            )

        self.confidence_bar.setValue(int(confidence * 100))
        self.detail_text.setPlainText(
            f"图像路径：{self.image_path}\n"
            f"预测类别：{class_name}\n"
            f"置信度：{confidence:.4f}\n"
            f"肺炎概率：{prob:.4f}\n"
            f"正常概率：{1 - prob:.4f}"
        )

    def _on_error(self, message):
        QMessageBox.critical(self, "检测失败", f"推理过程中出现错误：{message}")
        self.detail_text.setPlainText(f"检测失败：{message}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_image_display()


def main():
    app = QApplication(sys.argv)
    window = PneumoniaGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
