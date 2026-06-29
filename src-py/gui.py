import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
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

from inference import load_trained_model, predict


class PneumoniaGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("胸部 X 光肺炎检测系统")
        self.setMinimumSize(900, 500)

        self.model = None
        self.device = None
        self.image_path = None

        self._build_ui()
        self._load_model()

    def _build_ui(self):
        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
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

    def _load_model(self):
        try:
            self.model, self.device = load_trained_model()
            self.detail_text.setPlainText("模型加载成功。")
        except Exception as e:
            QMessageBox.critical(self, "模型加载失败", f"无法加载模型：{e}")

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择胸部 X 光图像", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not path:
            return

        self.image_path = path
        pixmap = QPixmap(path)
        scaled = pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)
        self.detect_btn.setEnabled(True)
        self.result_label.setText("尚未检测")
        self.result_label.setStyleSheet("font-size: 24px; color: gray;")
        self.confidence_bar.setValue(0)
        self.detail_text.setPlainText(f"已选择图像：{path}")

    def detect(self):
        if self.image_path is None or self.model is None:
            QMessageBox.warning(self, "提示", "请先选择图像并确保模型已加载。")
            return

        try:
            class_name, confidence, prob = predict(self.image_path, self.model, self.device)
            self.result_label.setText(class_name)
            if class_name == "PNEUMONIA":
                self.result_label.setStyleSheet("font-size: 24px; color: red; font-weight: bold;")
            else:
                self.result_label.setStyleSheet("font-size: 24px; color: green; font-weight: bold;")

            self.confidence_bar.setValue(int(confidence * 100))
            self.detail_text.setPlainText(
                f"图像路径：{self.image_path}\n"
                f"预测类别：{class_name}\n"
                f"置信度：{confidence:.4f}\n"
                f"肺炎概率：{prob:.4f}\n"
                f"正常概率：{1 - prob:.4f}"
            )
        except Exception as e:
            QMessageBox.critical(self, "检测失败", f"推理过程中出现错误：{e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image_label.pixmap():
            pixmap = self.image_label.pixmap()
            scaled = pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)


def main():
    app = QApplication(sys.argv)
    window = PneumoniaGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
