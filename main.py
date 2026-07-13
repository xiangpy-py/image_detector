#!/usr/bin/env python3
"""胸部 X 光肺炎检测系统 —— 项目入口。

使用方式:
    python main.py <command> [options]

命令:
    train       训练模型
    evaluate    评估模型并生成图表
    gui         启动图形界面
    cache       生成图像缓存（Rust）
    download    下载数据集
    dataset     管理数据集（add / list / remove / set）

环境变量配置示例:
    export DATASET_ROOT=/path/to/dataset
    export DATASET_ROOTS=/path/to/dataset1,/path/to/dataset2

也可直接通过 pip 安装后使用:
    pip install -e .
    image-detector gui
"""

import sys
from pathlib import Path

# 将 src-py 加入模块搜索路径
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "src-py"))

# 使用 importlib 显式导入 src-py/main.py，避免名称混淆与递归导入风险
import importlib.util

_src_main_path = _PROJECT_ROOT / "src-py" / "main.py"
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
        print("用法: python main.py <command> [options]")
        print()
        print("可用命令:")
        print("  train       训练模型")
        print("  evaluate    评估模型并生成图表")
        print("  gui         启动图形界面")
        print("  cache       使用 Rust 生成图像缓存")
        print("  download    下载数据集")
        print("  dataset     管理数据集（add / list / remove / set）")
        print()
        print("环境变量配置示例:")
        print('  export DATASET_ROOT=/path/to/dataset')
        print('  export DATASET_ROOTS=/path/to/dataset1,/path/to/dataset2')
        print()
        print('运行 "python main.py <command> --help" 查看各命令的详细选项。')
        sys.exit(0)

    main()
