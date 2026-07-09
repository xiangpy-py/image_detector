#!/usr/bin/env python3
"""胸部 X 光肺炎检测系统 —— 项目入口。

使用方式:
    python main.py <command> [options]

命令:
    train      训练模型
    evaluate   评估模型
    gui        启动图形界面
    cache      生成图像缓存（Rust）
    download   下载数据集

也可直接通过 pip 安装后使用:
    pip install -e .
    image-detector gui
"""

import sys
from pathlib import Path

# 将 src-py 加入模块搜索路径
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "src-py"))

from main import main

if __name__ == "__main__":
    main()
