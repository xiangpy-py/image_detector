import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from config import DEFAULT_THRESHOLD, OUTPUTS_DIR


def find_best_threshold(
    labels,
    probs,
    metric="f1",
    thresholds=None,
    min_recall=None,
    min_precision=None,
):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    labels = np.asarray(labels)
    probs = np.asarray(probs)

    best_threshold = DEFAULT_THRESHOLD
    best_score = -1.0

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(labels, preds, zero_division=0)
        elif metric == "precision":
            score = precision_score(labels, preds, zero_division=0)
        elif metric == "recall":
            score = recall_score(labels, preds, zero_division=0)
        elif metric == "f1_recall":
            recall = recall_score(labels, preds, zero_division=0)
            precision = precision_score(labels, preds, zero_division=0)
            score = 2 * precision * recall / (precision + recall + 1e-8)
            if min_recall is not None and recall < min_recall:
                continue
            if min_precision is not None and precision < min_precision:
                continue
        else:
            raise ValueError(f"不支持的优化指标: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, best_score


def save_threshold(threshold, path=None):
    if path is None:
        path = OUTPUTS_DIR / "threshold.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold}, f, indent=2)
    return path


def load_threshold(path=None):
    if path is None:
        path = OUTPUTS_DIR / "threshold.json"
    path = Path(path)
    if not path.exists():
        return DEFAULT_THRESHOLD
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("threshold", DEFAULT_THRESHOLD))
