import argparse
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CACHE_DIR,
    CACHE_SIZE,
    DATASET_ROOT,
    DATASET_ROOTS,
    EPOCHS,
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
        help="数据集根目录（默认读取环境变量 DATASET_ROOT）",
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
        help="图像缓存目录（默认读取环境变量 CACHE_DIR）",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="模型保存目录（默认读取环境变量 MODELS_DIR）",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="评估图表和日志输出目录（默认读取环境变量 OUTPUTS_DIR）",
    )
    parser.add_argument(
        "--app-data-dir",
        type=Path,
        default=None,
        help="应用数据根目录（设置后会自动推导 cache/models/outputs）",
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

    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.weight_decay is not None:
        overrides["weight_decay"] = args.weight_decay
    if args.patience is not None:
        overrides["patience"] = args.patience

    train(resume_from=args.resume, overrides=overrides or None)


def run_evaluate(args):
    _apply_path_args(args)
    from evaluate import run_evaluation

    run_evaluation(model_path=args.model_path, use_ema=not args.no_ema)


def run_gui(args):
    _apply_path_args(args)
    from gui import main as gui_main

    gui_main()


def run_cache(args):
    _apply_path_args(args)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from rust_preprocessor import preprocess_dataset as rust_preprocess
    except ImportError:
        logger.error(
            "Rust 扩展未安装。请先在项目根目录运行:\n"
            "  maturin develop    # 开发模式\n"
            "  maturin build      # 生产模式\n"
            "如果未安装 maturin: pip install maturin"
        )
        raise

    total_train = 0
    total_test = 0
    merged_info_path = CACHE_DIR / "merged_info.json"
    merged_info = {"datasets": []}

    datasets = DATASET_ROOTS
    if len(datasets) == 1:
        logger.info(
            f"开始 Rust 预处理: root={datasets[0]}, out={CACHE_DIR}, size={CACHE_SIZE}"
        )
    else:
        logger.info(
            f"开始 Rust 预处理: {len(datasets)} 个数据集, out={CACHE_DIR}, size={CACHE_SIZE}"
        )

    for i, root in enumerate(datasets):
        root = Path(root)
        if not root.exists():
            logger.error(f"数据集根目录不存在，跳过: {root}")
            continue

        # 子缓存目录名: 取路径最后一段，重名时加后缀
        base_name = root.name
        subdir_name = base_name
        counter = 1
        while (CACHE_DIR / subdir_name).exists():
            subdir_name = f"{base_name}_{counter}"
            counter += 1

        sub_cache = CACHE_DIR / subdir_name

        # 如果目标子缓存目录已存在且非空，且没有 --force，拒绝覆盖
        if sub_cache.exists() and any(sub_cache.iterdir()):
            if not args.force:
                logger.error(
                    f"目标缓存目录已存在且非空: {sub_cache}。"
                    "如需覆盖请加 --force，或先删除该目录。"
                )
                continue
            logger.warning(f"将覆盖已有缓存目录: {sub_cache}")

        sub_cache.mkdir(parents=True, exist_ok=True)

        if len(datasets) > 1:
            logger.info(
                f"[{i + 1}/{len(datasets)}] 处理数据集: {root} -> {subdir_name}"
            )
        else:
            logger.info(f"处理数据集: {root}")

        # Rust 预处理 -> 直接写到 CACHE_DIR 下（兼容单数据集时 flat cache）
        out_dir = CACHE_DIR if len(datasets) == 1 else sub_cache
        train_count, test_count = rust_preprocess(
            str(root), str(out_dir), CACHE_SIZE, True
        )
        total_train += train_count
        total_test += test_count

        # 多数据集时记录
        if len(datasets) > 1:
            merged_info["datasets"].append({
                "name": subdir_name,
                "root": str(root),
                "cache_subdir": str(sub_cache),
            })

    # 只有多数据集时才写 merged_info.json
    if len(datasets) > 1:
        import json
        with open(merged_info_path, "w", encoding="utf-8") as f:
            json.dump(merged_info, f, indent=2, ensure_ascii=False)
        logger.info(
            f"合并信息已保存: {merged_info_path} "
            f"({len(merged_info['datasets'])} 个数据集)"
        )

    logger.info(
        f"Rust 预处理完成: 共 {total_train} 张训练图像, {total_test} 张测试图像"
    )


def run_download(args):
    _apply_path_args(args)
    from download_database import download_dataset

    dataset_path = download_dataset(dataset_root=DATASET_ROOT)
    add_dataset("chest-xray-pneumonia", dataset_path)
    logger.info(f"数据集已注册为 'chest-xray-pneumonia': {dataset_path}")


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
    # 修复 autodl 等环境上 OMP_NUM_THREADS 设为空字符串导致的 libgomp 警告
    import os

    omp = os.environ.get("OMP_NUM_THREADS", "")
    if not omp.strip().isdigit():
        os.environ["OMP_NUM_THREADS"] = "1"

    if is_windows():
        import multiprocessing

        multiprocessing.set_start_method(get_multiprocessing_start_method(), force=True)

    setup_logger()

    parser = argparse.ArgumentParser(
        description="胸部 X 光肺炎检测系统",
        epilog=(
            "环境变量:\n"
            "  DATASET_ROOT      单个数据集根目录\n"
            "  DATASET_ROOTS     多个数据集根目录（逗号分隔，合并所有数据集）\n"
            "  CACHE_DIR         图像缓存目录（默认: ./cache）\n"
            "  MODELS_DIR        模型保存目录（默认: ./models）\n"
            "  OUTPUTS_DIR       评估图表和日志输出目录（默认: ./outputs）\n"
            "  KAGGLEHUB_CACHE   数据集下载缓存目录\n"
            "\n"
            "全局路径选项:\n"
            "  --dataset-root, --dataset-name, --cache-dir, --models-dir,\n"
            "  --outputs-dir, --app-data-dir  覆盖对应环境变量和默认值\n"
            "\n"
            "常用示例:\n"
            "  uv run main.py train\n"
            "  uv run main.py train --resume models/<timestamp>_best_model.pth\n"
            "  uv run main.py evaluate\n"
            "  uv run main.py cache --cache-dir /tmp/cache\n"
            "  uv run main.py download --dataset-root /path/to/chest_xray\n"
            "  uv run main.py dataset add chest1 /path/to/chest1\n"
            "  uv run main.py dataset list\n"
            "  uv run main.py dataset set chest1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", title="可用命令")

    train_parser = subparsers.add_parser(
        "train",
        help="训练肺炎检测模型",
        description="训练 DenseNet121 肺炎检测模型，支持从 checkpoint 恢复。",
    )
    train_parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="从指定 checkpoint (.pth) 恢复训练",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"训练轮数（默认读取 config.py 的 EPOCHS={EPOCHS}）",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=f"学习率（默认读取 config.py 的 LEARNING_RATE）",
    )
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="权重衰减系数（覆盖 config.py 的 WEIGHT_DECAY）",
    )
    train_parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="早停 patience（覆盖 config.py 的 EARLY_STOP_PATIENCE）",
    )
    _add_path_args(train_parser)

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="评估模型并生成图表",
        description="加载训练好的模型，在测试集上评估并输出 ROC、混淆矩阵等图表。",
    )
    eval_parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="指定要评估的模型文件路径（默认自动选最新的 *_best_model.pth）",
    )
    eval_parser.add_argument(
        "--no-ema",
        action="store_true",
        help="不使用 EMA 权重评估（默认会用，与训练时选 best 的口径一致）",
    )
    _add_path_args(eval_parser)

    gui_parser = subparsers.add_parser(
        "gui",
        help="启动图形界面",
        description="启动 PySide6 图形界面，支持单张图像推理。",
    )
    _add_path_args(gui_parser)

    cache_parser = subparsers.add_parser(
        "cache",
        help="生成图像缓存",
        description="使用 Rust 扩展预处理图像并生成 uint8 .npy 缓存（需先安装 Rust 扩展）。",
    )
    cache_parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已存在的缓存目录（默认拒绝覆盖）",
    )
    _add_path_args(cache_parser)

    download_parser = subparsers.add_parser(
        "download",
        help="下载数据集",
        description="从 Kaggle 下载胸部 X 光数据集并自动注册到数据集注册表。",
    )
    _add_path_args(download_parser)

    # ─── dataset 子命令 ───
    dataset_parser = subparsers.add_parser(
        "dataset",
        help="管理数据集注册表",
        description="添加、移除、列出、设置和查看已注册的数据集。",
        epilog=(
            "示例:\n"
            "  uv run main.py dataset add chest1 /path/to/chest1\n"
            "  uv run main.py dataset remove chest1\n"
            "  uv run main.py dataset list\n"
            "  uv run main.py dataset set chest1\n"
            "  uv run main.py dataset show chest1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    dataset_sub = dataset_parser.add_subparsers(dest="dataset_command", title="子命令")

    dataset_add = dataset_sub.add_parser("add", help="注册新数据集")
    dataset_add.add_argument("name", help="数据集名称")
    dataset_add.add_argument("path", help="数据集根目录路径")

    dataset_remove = dataset_sub.add_parser("remove", help="移除已注册数据集")
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
