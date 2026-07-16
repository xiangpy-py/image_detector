"""测试 threshold_tuner 模块。"""

import numpy as np
import pytest

from threshold_tuner import find_best_threshold


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
