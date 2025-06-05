import cv2
import numpy as np
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Iterable
import hashlib
import requests

REQUIRED_PACKAGES = [
    "numpy",
    "opencv-python",
    "scikit-image",
    "onnxruntime",
]


def check_and_install_packages(packages: Iterable[str] = REQUIRED_PACKAGES):
    for pkg in packages:
        if importlib.util.find_spec(pkg) is None:
            print(f"Package '{pkg}' fehlt. Installiere jetzt...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def ensure_package_version(pkg_name: str, required_version: str):
    try:
        import pkg_resources

        dist = pkg_resources.get_distribution(pkg_name)
        if dist.version != required_version:
            print(
                f"Warnung: '{pkg_name}' Version {dist.version} installiert, "
                f"aber {required_version} erforderlich."
            )
    except Exception:
        print(f"'{pkg_name}' nicht gefunden; Installation starten.")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", f"{pkg_name}=={required_version}"]
        )


def ensure_directory(path: str):
    dir_path = Path(path)
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Verzeichnis '{path}' konnte nicht erstellt werden: {e}")


def download_model_if_missing(model_url: str, save_path: str):
    path = Path(save_path)
    if path.exists():
        return
    print(f"Lade Modell von {model_url} ...")
    response = requests.get(model_url, stream=True)
    response.raise_for_status()
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def verify_checksum(file_path: str, expected_sha256: str) -> bool:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_sha256

def extract_contours(image_bin: np.ndarray) -> list:
    contours, _ = cv2.findContours(image_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    approx_contours = [cv2.approxPolyDP(cnt, epsilon=1.0, closed=False) for cnt in contours]
    return approx_contours
