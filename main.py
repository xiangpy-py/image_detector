#!/usr/bin/env python3

import sys
from pathlib import Path
import os

# PyInstaller 打包时会设置 _MEIPASS，指向临时解压目录
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # 打包模式：从临时目录加载模块
    _APP_DIR = Path(sys._MEIPASS)
    _PROJECT_ROOT = Path(sys.executable).parent
    _src_dir_name = "src_py"
else:
    # 开发模式：从源码目录加载
    _PROJECT_ROOT = Path(__file__).resolve().parent
    _APP_DIR = _PROJECT_ROOT
    _src_dir_name = "src-py"

sys.path.insert(0, str(_APP_DIR / _src_dir_name))

# 使用 importlib 显式导入 src-py/main.py，避免名称混淆与递归导入风险
import importlib.util

_src_main_path = _APP_DIR / _src_dir_name / "main.py"
_spec = importlib.util.spec_from_file_location("_src_main", _src_main_path)
_src_main = importlib.util.module_from_spec(_spec)
sys.modules["_src_main"] = _src_main
_spec.loader.exec_module(_src_main)
main = _src_main.main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 无参数时显示帮助信息
        print("胸部 X 光肺炎检测系统")
        print()
        print("用法: uv run main.py <command> [options]")
        print()
        print("可用命令:")
        print("  train       训练肺炎检测模型")
        print("  evaluate    评估模型并生成图表")
        print("  gui         启动图形界面")
        print("  cache       使用 Rust 生成图像缓存")
        print("  download    下载数据集")
        print("  dataset     管理数据集注册表")
        print()
        print("环境变量:")
        print("  DATASET_ROOT      单个数据集根目录")
        print("  DATASET_ROOTS     多个数据集根目录（逗号分隔，合并所有数据集）")
        print("  CACHE_DIR         图像缓存目录（默认: ./cache）")
        print("  MODELS_DIR        模型保存目录（默认: ./models）")
        print("  OUTPUTS_DIR       评估图表和日志输出目录（默认: ./outputs）")
        print("  KAGGLEHUB_CACHE   数据集下载缓存目录")
        print()
        print("全局路径选项:")
        print("  --dataset-root, --dataset-name, --cache-dir, --models-dir,")
        print("  --outputs-dir, --app-data-dir  覆盖对应环境变量和默认值")
        print()
        print("示例:")
        print("  uv run main.py train")
        print("  uv run main.py train --resume models/<timestamp>_best_model.pth")
        print("  uv run main.py evaluate")
        print("  export DATASET_ROOT=/path/to/dataset && uv run main.py train")
        print("  uv run main.py cache --cache-dir /tmp/cache")
        print("  uv run main.py download --dataset-root /path/to/chest_xray")
        print("  uv run main.py dataset add chest1 /path/to/chest1")
        print("  uv run main.py dataset list")
        print("  uv run main.py dataset set chest1")
        print()
        print('运行 "uv run main.py <command> --help" 查看各命令的详细选项。')
        sys.exit(0)

    main()
