"""模型文件查找工具。

统一在 MODELS_DIR 下查找带时间戳的最佳/最新模型，避免在多处复制查找逻辑。
"""

from pathlib import Path

from config import MODELS_DIR


def find_model_path(model_path: Path | str | None = None) -> Path:
    """查找要加载的模型文件。

    优先级：
    1. 显式传入的 model_path
    2. 带时间戳前缀的 *_best_model.pth（按文件名排序取最新）
    3. 带时间戳前缀的 *_best_auc_model.pth
    4. 带时间戳前缀的 *_last_model.pth
    5. 旧版 best_model.pth（兼容历史文件）

    Args:
        model_path: 用户显式指定的模型路径，None 则按优先级自动选择

    Returns:
        Path: 解析后的模型文件绝对路径

    Raises:
        FileNotFoundError: 未找到任何模型文件
    """
    if model_path is not None:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"指定的模型文件不存在: {p}")
        return p.resolve()

    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"模型目录不存在: {MODELS_DIR}")

    patterns = [
        "*_best_model.pth",
        "*_best_auc_model.pth",
        "*_last_model.pth",
    ]
    for pattern in patterns:
        matches = sorted(MODELS_DIR.glob(pattern), reverse=True)
        if matches:
            return matches[0]

    legacy = MODELS_DIR / "best_model.pth"
    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        f"在 {MODELS_DIR} 中未找到任何模型文件，请先运行 `uv run main.py train`。"
    )
