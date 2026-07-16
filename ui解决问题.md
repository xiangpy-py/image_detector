以下是针对你当前 PySide6 + Linux 环境下文件对话框显示异常的**完整工程级解决方案**，可直接按步骤落地。

---

## 一、问题定性

| 现象 | 根因 |
|------|------|
| 对话框全白/空白，文字不可见 | `QWidget { background-color: #F8FAFC; }` 全局样式泄漏到 `QFileDialog`，与系统 GTK 主题冲突 |
| 只有点击才泛蓝显示文字 | 选中项的强制高亮（`QPalette::Highlight`）覆盖了背景，暂时暴露文字 |
| 字体发虚或不对 | Linux 缺少 `"Microsoft YaHei"` 等 Windows 字体，且未配置回退栈 |
| 左侧目录树异常 | 使用了系统原生对话框（GTK），在 Qt 样式表干预下渲染不一致 |

---

## 二、修复步骤（按优先级排序）

### 步骤 1：强制使用 Qt 内置对话框（立即见效）

**文件：** `gui.py`  
**定位：** `_file_dialog_options()` 函数

**原代码（问题）：**
```python
def _file_dialog_options() -> QFileDialog.Option:
    if sys.platform == "win32":
        return QFileDialog.Option(0)

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not has_display:
        return QFileDialog.Option.DontUseNativeDialog

    return QFileDialog.Option(0)   # ← Linux 有桌面时走原生 GTK，与 QSS 冲突
```

**修复后：**
```python
def _file_dialog_options() -> QFileDialog.Option:
    """
    返回文件对话框选项。
    Linux/macOS 强制使用 Qt 内置对话框，避免 GTK/QSS 样式冲突。
    Windows 使用原生对话框以获得更好体验。
    """
    if sys.platform == "win32":
        return QFileDialog.Option(0)
    # Linux / macOS：一律非原生，确保颜色、字体完全受控于 QSS
    return QFileDialog.Option.DontUseNativeDialog
```

> **验证：** 修改后重启程序，点击「选择图像」，对话框应变为 Qt 经典风格（白色背景、蓝色标题栏、左侧目录树正常），文字立即可见。

---

### 步骤 2：隔离全局样式表，防止污染弹窗

**文件：** `gui.py`  
**定位：** `_build_ui()` 末尾的 `self.setStyleSheet(...)` 

**问题代码段：**
```python
self.setStyleSheet("""
    QWidget {
        font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
        background-color: #F8FAFC;   # ← 这行会污染所有 QWidget 子类，包括 QFileDialog
    }
    ...
""")
```

**修复策略：** 将 `QWidget` 的全局背景移除，改为仅对主窗口类生效。

**修复后：**
```python
self.setStyleSheet("""
    /* ========== 1. 仅主窗口容器设背景，不污染子对话框 ========== */
    PneumoniaGUI {
        background-color: #F8FAFC;
    }

    /* ========== 2. 字体栈：Linux 优先回退到系统自带中文字体 ========== */
    QWidget {
        font-family: "Segoe UI", "Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans", sans-serif;
        color: #1E293B;
    }

    /* ========== 3. 输入控件 ========== */
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

    /* ========== 4. 日志区（暗色） ========== */
    QTextEdit {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        background: #0F172A;
        color: #E2E8F0;
        font-family: "JetBrains Mono", "Consolas", "Courier New", monospace;
        font-size: 12px;
        padding: 8px;
    }

    /* ========== 5. 表格 ========== */
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

    /* ========== 6. 按钮基础 ========== */
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

    /* ========== 7. 分组框（你的 _style_group 已设，这里兜底） ========== */
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

    /* ========== 8. 复选框 ========== */
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

    /* ========== 9. 进度条 ========== */
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

    /* ========== 10. 对话框保险（万一仍有原生对话框被唤起） ========== */
    QDialog, QFileDialog, QMessageBox {
        background-color: #FFFFFF;
        color: #1E293B;
    }
    QDialog QLabel, QFileDialog QLabel, QMessageBox QLabel {
        color: #1E293B;
    }
    QDialog QPushButton, QFileDialog QPushButton, QMessageBox QPushButton {
        background-color: #3B82F6;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
    }
""")
```

---

### 步骤 3：确保 Linux 中文字体已安装

即使代码里写了字体回退，如果系统完全缺失中文字体，Qt 会回退到无法显示的字体。

**检查命令：**
```bash
fc-list :lang=zh | grep -i "noto\|wenquanyi\|microhei"
```

**如未安装，按需安装：**

| 发行版 | 命令 |
|--------|------|
| Ubuntu/Debian | `sudo apt install fonts-noto-cjk fonts-wqy-microhei` |
| Fedora | `sudo dnf install google-noto-sans-cjk-fonts wqy-microhei-fonts` |
| Arch | `sudo pacman -S noto-fonts-cjk wqy-microhei` |

安装后**重启程序**（无需重启系统，但需重启 Qt 应用以重新加载字体缓存）。

---

### 步骤 4：入口函数增加高分屏与字体渲染优化

**文件：** `gui.py`  
**定位：** `main()` 函数

**修复后：**
```python
def main():
    # 高分屏适配
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # 字体抗锯齿
    QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL, False)
    
    app = QApplication(sys.argv)
    
    # 统一应用级字体，作为最后回退
    font = app.font()
    font.setFamily("Noto Sans CJK SC")  # Linux 最稳妥的跨发行版中文字体
    font.setPointSize(10)
    app.setFont(font)
    
    window = PneumoniaGUI()
    window.show()
    sys.exit(app.exec())
```

---

### 步骤 5：清理 `_style_group` 与 `_style_button` 的重复样式

你代码中既有函数动态设置样式，又有全局 `setStyleSheet`。两者冲突时，**后加载的优先级更高**，但具体表现取决于选择器权重。

**建议：** 保留全局样式表作为"主题"，`_style_button` 仅用于动态变色（如成功/警告/危险色），`_style_group` 不再重复设置 `background/border`（因为全局已设）。

**修改 `_style_group`（仅保留标题相关）：**
```python
def _style_group(title):
    """统一分组框样式——现代卡片风格（仅标题和边距，颜色由全局 QSS 控制）"""
    g = QGroupBox(title)
    g.setStyleSheet("""
        QGroupBox {
            font-weight: bold;
            font-size: 14px;
            margin-top: 12px;
            padding-top: 16px;
            padding-left: 14px;
            padding-right: 14px;
            padding-bottom: 14px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 14px;
            padding: 0 8px;
            color: #475569;
            font-size: 13px;
        }
    """)
    return g
```

---

## 三、验证清单

修改完成后，按以下顺序验证：

1. **启动程序** → 主界面背景应为 `#F8FAFC` 浅灰蓝，文字清晰。
2. **点击「选择图像」** → 弹出对话框背景应为白色/浅灰，左侧目录树可见，文件列表文字清晰，无需点击即可阅读。
3. **切换目录** → 无闪烁、无残影。
4. **点击「批量检测」选择文件夹** → `QFileDialog.getExistingDirectory` 同样表现正常。
5. **触发 QMessageBox**（如未加载模型时点击检测）→ 弹窗文字清晰，按钮样式正常。
6. **调整窗口大小** → 图像预览区自适应，无布局崩坏。

---

## 四、深层原因与预防规范

### 为什么 Linux 比 Windows 更容易出现此问题？

| 维度 | Windows | Linux |
|------|---------|-------|
| 对话框实现 | Win32 API，Qt 能较好隔离样式 | 默认走 GTK3/Portal，Qt 通过 `QGtkStyle` 桥接，QSS 极易穿透 |
| 字体回退 | 系统自带微软雅黑、宋体 | 需手动安装 Noto/WenQuanYi，否则回退到无中文覆盖的字体 |
| 颜色主题 | Qt 与 Win32 渲染分离 | GTK 主题（如 Adwaita、Breeze）会强制覆盖部分 Qt 控件颜色 |

### 工程规范（预防复发）

1. **永不使用 `QWidget { background-color: ... }` 全局设置**  
   只针对具体类名（如 `PneumoniaGUI`）或具体控件（如 `QTextEdit`）设置背景。

2. **Linux 下 QFileDialog 默认非原生**  
   封装一个统一的 `AppFileDialog` 类，内部强制 `DontUseNativeDialog`，全项目复用。

3. **字体栈必须包含 Linux 回退**  
   规范格式：`"西文优先", "中文优先", "Linux 中文回退", "通用回退"`。

4. **样式与布局分离**  
   建议将 QSS 抽离到独立文件（如 `resources/style.qss`），通过 `QFile` 加载，便于调试和主题切换。

---

## 五、备选方案（如果上述方案仍不理想）

如果因为某种原因**必须使用系统原生对话框**（例如需要文件预览、网络挂载等），则**完全禁用 QSS 对对话框的干预**：

```python
# 在弹出对话框前临时清空样式，关闭后恢复
def select_image_native(self):
    # 保存当前样式
    old_style = self.styleSheet()
    # 移除对话框相关样式（或全部样式）
    self.setStyleSheet("")  
    
    path, _ = QFileDialog.getOpenFileName(
        self, "选择胸部 X 光图像", "", "Images (*.png *.jpg *.jpeg)"
    )
    
    # 恢复样式
    self.setStyleSheet(old_style)
    
    if path:
        self.image_path = path
        # ...
```

> **不推荐**作为默认方案，因为会导致主界面闪烁且破坏一致性，仅作为极端兼容手段。

---

## 六、总结

你当前的问题由**三个因素叠加**导致：
1. Linux 下 `QFileDialog` 默认走 GTK 原生，与 Qt 样式表冲突；
2. 全局 `QWidget { background-color: ... }` 污染了对话框的背景和文字颜色；
3. 字体回退栈未覆盖 Linux 系统字体。

**核心修复只需两步：**
1. `_file_dialog_options()` 返回 `QFileDialog.Option.DontUseNativeDialog`；
2. `setStyleSheet()` 中把 `QWidget` 的全局 `background-color` 改为 `PneumoniaGUI` 限定。

按上述步骤修改后，对话框应恢复正常显示。如果仍有问题，请提供你的 Linux 发行版（如 Ubuntu 22.04/Fedora 40）和桌面环境（GNOME/KDE），可进一步针对性调整。