import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["SKIZZE_SKIP_INSTALL"] = "1"
from skizze.utils import ensure_directory


def test_ensure_directory(tmp_path):
    path = tmp_path / "subdir"
    ensure_directory(str(path))
    assert path.exists()
