"""Pytest 配置与 fixtures。"""

import sys
from pathlib import Path

# 确保 src-py 在 PYTHONPATH 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src-py"))
