#!/usr/bin/env python3

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
        print("用法: uv run main.py <command> [options]")
        print()
        print("可用命令:")
        print("  train       训练模型")
        print("  evaluate    评估模型并生成图表")
        print("  gui         启动图形界面")
        print("  cache       使用 Rust 生成图像缓存")
        print("  download    下载数据集")
        print("  dataset     管理数据集（add / list / remove / set）")
        print()
        print("环境变量:")
        print("  DATASET_ROOT      单个数据集根目录")
        print("  DATASET_ROOTS     多个数据集根目录（逗号分隔，取第一个）")
        print("  CACHE_DIR         图像缓存目录（默认: 项目根目录/cache）")
        print("  MODELS_DIR        模型保存目录")
        print("  OUTPUTS_DIR       评估图表和日志输出目录")
        print("  KAGGLEHUB_CACHE   数据集下载缓存目录")
        print()
        print("示例:")
        print("  uv run main.py train")
        print("  export DATASET_ROOT=/path/to/dataset && uv run main.py train")
        print("  set DATASET_ROOT=C:\\datasets && uv run main.py download")
        print("  uv run main.py cache --cache-dir D:\\cache")
        print()
        print('运行 "uv run main.py <command> --help" 查看各命令的详细选项。')
        sys.exit(0)

    main()
