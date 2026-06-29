import argparse
import subprocess
import sys
from pathlib import Path

from config import CACHE_DIR, DATASET_ROOT


def run_train():
    from train import train

    train()


def run_evaluate():
    from evaluate import run_evaluation

    run_evaluation()


def run_gui():
    from gui import main as gui_main

    gui_main()


def run_cache():
    binary = Path("target/release/preprocess")
    if not binary.exists():
        print("未找到 Rust 预处理二进制文件，尝试编译...")
        subprocess.run(["cargo", "build", "--release", "--bin", "preprocess"], check=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(binary),
            "--root",
            str(DATASET_ROOT),
            "--out",
            str(CACHE_DIR),
            "--size",
            "224",
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description="胸部 X 光肺炎检测系统")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="训练模型")
    subparsers.add_parser("evaluate", help="评估模型并生成图表")
    subparsers.add_parser("gui", help="启动 GUI")
    subparsers.add_parser("cache", help="使用 Rust 生成图像缓存")

    args = parser.parse_args()

    commands = {
        "train": run_train,
        "evaluate": run_evaluate,
        "gui": run_gui,
        "cache": run_cache,
    }

    commands[args.command]()


if __name__ == "__main__":
    main()
