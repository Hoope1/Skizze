import os
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["SKIZZE_SKIP_VENV"] = "1"
os.environ["SKIZZE_SKIP_INSTALL"] = "1"

from skizze import cli


def create_test_image(tmp_path):
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.line(img, (5, 5), (45, 45), (255, 255, 255), 2)
    path = tmp_path / "test.png"
    cv2.imwrite(str(path), img)
    return path


def test_process_image_basic(tmp_path):
    img_path = create_test_image(tmp_path)
    result = cli.process_image(str(img_path), output_dir=None)
    assert set(result.keys()) == {
        "binary",
        "edges",
        "lines",
        "contours",
        "combined",
        "threshold_value",
    }
    assert result["binary"].shape == (50, 50)

