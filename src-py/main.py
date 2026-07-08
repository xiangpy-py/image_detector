import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CACHE_DIR,
    DATASET_ROOT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    override_paths,
)
from system import get_multiprocessing_start_method, is_windows


def _add_path_args(parser):
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="数据集根目录（默认从环境变量 DATASET_ROOT 或平台默认值获取）",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="缓存目录（默认从环境变量 CACHE_DIR 或 ./cache 获取）",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="模型保存目录（默认从环境变量 MODELS_DIR 或 ./models 获取）",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="输出目录（默认从环境变量 OUTPUTS_DIR 或 ./outputs 获取）",
    )


def _apply_path_args(args):
    override_paths(
        dataset_root=args.dataset_root,
        cache_dir=args.cache_dir,
        models_dir=args.models_dir,
        outputs_dir=args.outputs_dir,
    )


def run_train(args):
    _apply_path_args(args)
    from train import train

    train()


def run_evaluate(args):
    _apply_path_args(args)
    from evaluate import run_evaluation

    run_evaluation()


def run_gui(args):
    _apply_path_args(args)
    from gui import main as gui_main

    gui_main()


def run_cache(args):
    _apply_path_args(args)
    binary = Path("target/release/preprocess")
    if not binary.exists():
        print("未找到 Rust 预处理二进制文件，尝试编译...")
        subprocess.run(["cargo", "build", "--release", "--bin", "preprocess"], check=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    mean_str = ",".join(str(v) for v in IMAGENET_MEAN)
    std_str = ",".join(str(v) for v in IMAGENET_STD)

    cmd = [
        str(binary),
        "--root",
        str(DATASET_ROOT),
        "--out",
        str(CACHE_DIR),
        "--size",
        str(IMG_SIZE),
        "--mean",
        mean_str,
        "--std",
        std_str,
    ]

    if is_windows():
        subprocess.run(cmd, check=True, shell=False)
    else:
        subprocess.run(cmd, check=True)


def run_download(args):
    _apply_path_args(args)
    from download_database import download_dataset

    download_dataset()


def main():
    if is_windows():
        import multiprocessing

        multiprocessing.set_start_method(get_multiprocessing_start_method(), force=True)

    parser = argparse.ArgumentParser(description="胸部 X 光肺炎检测系统")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练模型")
    _add_path_args(train_parser)

    eval_parser = subparsers.add_parser("evaluate", help="评估模型并生成图表")
    _add_path_args(eval_parser)

    gui_parser = subparsers.add_parser("gui", help="启动 GUI")
    _add_path_args(gui_parser)

    cache_parser = subparsers.add_parser("cache", help="使用 Rust 生成图像缓存")
    _add_path_args(cache_parser)

    download_parser = subparsers.add_parser("download", help="下载数据集")
    _add_path_args(download_parser)

    args = parser.parse_args()

    commands = {
        "train": run_train,
        "evaluate": run_evaluate,
        "gui": run_gui,
        "cache": run_cache,
        "download": run_download,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
