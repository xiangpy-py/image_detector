import argparse
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CACHE_DIR,
    CACHE_SIZE,
    DATASET_ROOT,
    override_paths,
)
from logger_config import setup_logger
from system import (
    add_dataset,
    get_active_dataset,
    get_dataset_path,
    get_multiprocessing_start_method,
    is_windows,
    list_datasets,
    load_dataset_registry,
    remove_dataset,
    save_dataset_registry,
    set_active_dataset,
)


def _add_path_args(parser):
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="数据集根目录（默认从环境变量 DATASET_ROOT 或平台默认值获取）",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="已注册的数据集名称（与 --dataset-root 二选一，优先级更高）",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="缓存目录（默认从环境变量 CACHE_DIR 或应用数据目录获取）",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="模型保存目录（默认从环境变量 MODELS_DIR 或应用数据目录获取）",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="输出目录（默认从环境变量 OUTPUTS_DIR 或应用数据目录获取）",
    )
    parser.add_argument(
        "--app-data-dir",
        type=Path,
        default=None,
        help="应用数据目录（设置后会自动推导 cache/models/outputs）",
    )


def _apply_path_args(args):
    override_paths(
        dataset_root=args.dataset_root,
        cache_dir=args.cache_dir,
        models_dir=args.models_dir,
        outputs_dir=args.outputs_dir,
        app_data_dir=getattr(args, "app_data_dir", None),
        dataset_name=getattr(args, "dataset_name", None),
    )


def run_train(args):
    _apply_path_args(args)
    from train import train

    train(resume_from=args.resume)


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

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from rust_preprocessor import preprocess_dataset
    except ImportError:
        logger.error(
            "Rust 扩展未安装。请先在项目根目录运行:\n"
            "  maturin develop    # 开发模式\n"
            "  maturin build      # 生产模式\n"
            "如果未安装 maturin: pip install maturin"
        )
        raise

    logger.info(
        f"开始 Rust 预处理: root={DATASET_ROOT}, out={CACHE_DIR}, size={CACHE_SIZE}"
    )
    train_count, test_count = preprocess_dataset(
        str(DATASET_ROOT), str(CACHE_DIR), CACHE_SIZE
    )
    logger.info(f"Rust 预处理完成: train={train_count}, test={test_count}")


def run_download(args):
    _apply_path_args(args)
    from download_database import download_dataset

    download_dataset()


def run_dataset(args):
    """数据集管理子命令。"""
    if args.dataset_command == "add":
        path = Path(args.path)
        if not path.exists():
            logger.error(f"路径不存在: {path}")
            return
        add_dataset(args.name, path)
        logger.info(f"数据集 '{args.name}' 已添加: {path}")
    elif args.dataset_command == "remove":
        try:
            remove_dataset(args.name)
            logger.info(f"数据集 '{args.name}' 已移除")
        except KeyError as e:
            logger.error(str(e))
    elif args.dataset_command == "list":
        reg = load_dataset_registry()
        datasets = reg.get("datasets", {})
        active = reg.get("active")
        if not datasets:
            logger.info("暂无注册的数据集")
            return
        logger.info("已注册的数据集:")
        for name, info in datasets.items():
            marker = " ★" if name == active else ""
            logger.info(f"  {name}: {info['path']}{marker}")
        if active:
            logger.info(f"当前默认数据集: {active}")
    elif args.dataset_command == "set":
        try:
            set_active_dataset(args.name)
            logger.info(f"默认数据集已设置为 '{args.name}'")
        except KeyError as e:
            logger.error(str(e))
    elif args.dataset_command == "show":
        try:
            path = get_dataset_path(args.name)
            logger.info(f"数据集 '{args.name}': {path}")
        except KeyError as e:
            logger.error(str(e))


def main():
    if is_windows():
        import multiprocessing

        multiprocessing.set_start_method(get_multiprocessing_start_method(), force=True)

    setup_logger()

    parser = argparse.ArgumentParser(
        description="胸部 X 光肺炎检测系统",
        epilog=(
            "环境变量:\n"
            "  DATASET_ROOT      单个数据集根目录\n"
            "  DATASET_ROOTS     多个数据集根目录（逗号分隔）\n"
            "  CACHE_DIR         缓存目录\n"
            "  MODELS_DIR        模型保存目录\n"
            "  OUTPUTS_DIR       输出目录\n"
            "\n示例:\n"
            "  python main.py train --dataset-name chest1\n"
            "  export DATASET_ROOT=/path/to/dataset && python main.py train\n"
            "  python main.py dataset add chest1 /path/to/chest1\n"
            "  python main.py dataset list"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="从指定 checkpoint 恢复训练",
    )
    _add_path_args(train_parser)

    eval_parser = subparsers.add_parser("evaluate", help="评估模型并生成图表")
    _add_path_args(eval_parser)

    gui_parser = subparsers.add_parser("gui", help="启动 GUI")
    _add_path_args(gui_parser)

    cache_parser = subparsers.add_parser("cache", help="使用 Rust 生成图像缓存")
    _add_path_args(cache_parser)

    download_parser = subparsers.add_parser("download", help="下载数据集")
    _add_path_args(download_parser)

    # ─── dataset 子命令 ───
    dataset_parser = subparsers.add_parser("dataset", help="管理数据集注册表")
    dataset_sub = dataset_parser.add_subparsers(dest="dataset_command")

    dataset_add = dataset_sub.add_parser("add", help="注册数据集")
    dataset_add.add_argument("name", help="数据集名称")
    dataset_add.add_argument("path", help="数据集根目录路径")

    dataset_remove = dataset_sub.add_parser("remove", help="移除数据集")
    dataset_remove.add_argument("name", help="数据集名称")

    dataset_list = dataset_sub.add_parser("list", help="列出已注册数据集")

    dataset_set = dataset_sub.add_parser("set", help="设置默认数据集")
    dataset_set.add_argument("name", help="数据集名称")

    dataset_show = dataset_sub.add_parser("show", help="查看数据集路径")
    dataset_show.add_argument("name", help="数据集名称")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "train": run_train,
        "evaluate": run_evaluate,
        "gui": run_gui,
        "cache": run_cache,
        "download": run_download,
        "dataset": run_dataset,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
