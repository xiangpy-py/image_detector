"""测试 system 模块。"""

import os
from pathlib import Path

import pytest

from system import (
    get_app_data_dir,
    get_default_cache_dir,
    get_default_dataset_root,
    get_default_kaggle_cache_dir,
    get_default_models_dir,
    get_default_outputs_dir,
    get_multiprocessing_start_method,
    get_num_workers_default,
    is_linux,
    is_macos,
    is_windows,
    normalize_path,
    set_app_data_dir,
)


def test_is_functions_return_bool():
    assert isinstance(is_linux(), bool)
    assert isinstance(is_windows(), bool)
    assert isinstance(is_macos(), bool)


def test_get_num_workers_default():
    n = get_num_workers_default()
    assert isinstance(n, int)
    assert n >= 0
    assert n <= 4


def test_get_multiprocessing_start_method():
    method = get_multiprocessing_start_method()
    assert method in ("spawn", "fork")


def test_normalize_path():
    path = normalize_path(Path("."))
    assert path.is_absolute()
    assert path.exists()


def test_get_app_data_dir():
    app_dir = get_app_data_dir()
    assert isinstance(app_dir, Path)
    assert app_dir.is_absolute()


def test_set_app_data_dir():
    original = get_app_data_dir()
    test_path = Path("/tmp/test_pneumonia_dir")
    set_app_data_dir(test_path)
    assert get_app_data_dir() == test_path
    # 恢复
    set_app_data_dir(original)


def test_get_default_cache_dir():
    cache = get_default_cache_dir()
    assert isinstance(cache, Path)
    assert cache.is_absolute()


def test_get_default_models_dir():
    models = get_default_models_dir()
    assert isinstance(models, Path)
    assert models.is_absolute()


def test_get_default_outputs_dir():
    outputs = get_default_outputs_dir()
    assert isinstance(outputs, Path)
    assert outputs.is_absolute()


def test_get_default_dataset_root_env_override(monkeypatch):
    monkeypatch.setenv("DATASET_ROOT", "/custom/dataset")
    root = get_default_dataset_root()
    assert str(root) == "/custom/dataset"


def test_get_default_kaggle_cache_dir_env_override(monkeypatch):
    monkeypatch.setenv("KAGGLEHUB_CACHE", "/custom/kaggle")
    cache = get_default_kaggle_cache_dir()
    assert str(cache) == "/custom/kaggle"


def test_get_default_cache_dir_env_override(monkeypatch):
    monkeypatch.setenv("CACHE_DIR", "/custom/cache")
    cache = get_default_cache_dir()
    assert str(cache) == "/custom/cache"


def test_get_default_models_dir_env_override(monkeypatch):
    monkeypatch.setenv("MODELS_DIR", "/custom/models")
    models = get_default_models_dir()
    assert str(models) == "/custom/models"
