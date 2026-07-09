"""测试 config 模块。"""

from pathlib import Path

from config import CACHE_SIZE, CLASS_NAMES, IMG_SIZE, LABEL_MAP, override_paths


def test_class_names():
    assert CLASS_NAMES == ["NORMAL", "PNEUMONIA"]


def test_label_map():
    assert LABEL_MAP["NORMAL"] == 0
    assert LABEL_MAP["PNEUMONIA"] == 1


def test_img_size():
    assert IMG_SIZE == 224


def test_cache_size():
    assert CACHE_SIZE == 256


def test_override_paths():
    """测试 override_paths 能正确修改全局路径。"""
    import config

    original_root = config.DATASET_ROOT
    test_path = Path("/tmp/test_dataset")

    override_paths(dataset_root=test_path)
    assert config.DATASET_ROOT == test_path

    # 恢复
    override_paths(dataset_root=original_root)
