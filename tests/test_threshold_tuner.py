"""测试 threshold_tuner 模块。"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from threshold_tuner import (
    DEFAULT_THRESHOLD,
    find_best_threshold,
    load_threshold,
    save_threshold,
)


def test_find_best_threshold_f1():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    threshold, score = find_best_threshold(labels, probs, metric="f1")
    assert 0 <= threshold <= 1
    assert 0 <= score <= 1


def test_find_best_threshold_perfect():
    """完美分离时，最优阈值应在中间。"""
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    threshold, score = find_best_threshold(labels, probs, metric="f1")
    assert pytest.approx(score, 0.01) == 1.0


def test_find_best_threshold_precision():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    threshold, score = find_best_threshold(labels, probs, metric="precision")
    assert 0 <= threshold <= 1
    assert 0 <= score <= 1


def test_find_best_threshold_recall():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    threshold, score = find_best_threshold(labels, probs, metric="recall")
    assert 0 <= threshold <= 1
    assert 0 <= score <= 1


def test_find_best_threshold_invalid_metric():
    with pytest.raises(ValueError):
        find_best_threshold(np.array([0, 1]), np.array([0.5, 0.5]), metric="invalid")


def test_save_and_load_threshold():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "threshold.json"
        save_threshold(0.73, path=path)
        loaded = load_threshold(path=path)
        assert pytest.approx(loaded, 0.001) == 0.73


def test_load_threshold_missing():
    """文件不存在时应返回默认值 0.5。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nonexistent.json"
        loaded = load_threshold(path=path)
        assert loaded == DEFAULT_THRESHOLD


def test_save_threshold_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "deep" / "nested" / "threshold.json"
        returned_path = save_threshold(0.42, path=path)
        assert returned_path.exists()
        with open(returned_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["threshold"] == 0.42
