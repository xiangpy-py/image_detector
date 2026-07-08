"""测试 system 模块。"""

import os
from pathlib import Path

import pytest

from system import (
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


def test_get_default_cache_dir():
    root = Path("/tmp/fake_project")
    cache = get_default_cache_dir(root)
    assert cache == root / "cache"


def test_get_default_models_dir():
    root = Path("/tmp/fake_project")
    models = get_default_models_dir(root)
    assert models == root / "models"


def test_get_default_outputs_dir():
    root = Path("/tmp/fake_project")
    outputs = get_default_outputs_dir(root)
    assert outputs == root / "outputs"


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
    root = Path("/tmp/fake_project")
    cache = get_default_cache_dir(root)
    assert str(cache) == "/custom/cache"


def test_get_default_models_dir_env_override(monkeypatch):
    monkeypatch.setenv("MODELS_DIR", "/custom/models")
    root = Path("/tmp/fake_project")
    models = get_default_models_dir(root)
    assert str(models) == "/custom/models"
