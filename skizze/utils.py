import os
import sys
import subprocess
from pathlib import Path
import venv
import hashlib
import logging
import requests

# 12.1 Virtual environment
def ensure_venv():
    project_root = Path(__file__).parent.parent.resolve()
    venv_dir = project_root / "env"
    if str(sys.prefix).startswith(str(venv_dir)):
        return
    if not venv_dir.exists():
        print("🔧 Creating virtual environment in './env' …")
        venv.EnvBuilder(with_pip=True).create(str(venv_dir))
    python_exe = venv_dir / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    os.execv(str(python_exe), [str(python_exe)] + sys.argv)

# 12.2 Required packages list
REQUIRED_PACKAGES = [
    "numpy",
    "opencv-python",
    "scikit-image",
    "onnxruntime",
    "torch",
    "requests",
    "toml"
]

def install_missing_packages(packages):
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"📦 Package '{pkg}' missing. Installing …")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# 12.3 SHA256 verification
def verify_checksum(file_path: str, expected_sha256: str) -> bool:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest().lower() == expected_sha256.lower()

# 12.4 Model download utility
def download_model_if_missing(model_url: str, save_path: str, expected_sha256: str = None):
    path = Path(save_path)
    if path.parent == Path("") or not path.parent.exists():
        model_dir = Path("models")
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / path.name
    if path.exists():
        if expected_sha256 and verify_checksum(str(path), expected_sha256):
            print(f"✅ Checksum ok for '{path}'.")
            return
        elif expected_sha256:
            print(f"❌ Checksum mismatch for '{path}'. Re‑downloading.")
            path.unlink()
        else:
            return
    print(f"🔽 Downloading model from '{model_url}' → '{path}'")
    response = requests.get(model_url, stream=True)
    response.raise_for_status()
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    if expected_sha256 and not verify_checksum(str(path), expected_sha256):
        raise RuntimeError("Checksum mismatch after download!")

# 12.6 Logging setup
def setup_logging(level: str = "INFO", log_file: str = None):
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    numeric = getattr(logging, level.upper(), logging.INFO)
    if log_file:
        logging.basicConfig(level=numeric, format=fmt, filename=log_file, filemode="a")
    else:
        logging.basicConfig(level=numeric, format=fmt)
