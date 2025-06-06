import sys
from pathlib import Path

from .utils import ensure_venv, install_missing_packages, REQUIRED_PACKAGES, setup_logging

ensure_venv()
install_missing_packages(REQUIRED_PACKAGES)

import argparse
import logging
import os
import numpy as np
import cv2
import torch
from .config import load_config_from_pyproject
from .preprocessing import load_image, to_grayscale, denoise_image, morphological_opening, morphological_closing
from .thresholding import auto_threshold_otsu, adaptive_threshold
from .drawing import draw_lines_on_blank, combine_line_images
from .line_detection import detect_edges_canny, detect_lines_hough, filter_lines_by_length
setup_logging()
log = logging.getLogger(__name__)
cfg = load_config_from_pyproject()

def process_image(path, output_dir=None):
    img = load_image(path)
    gray = to_grayscale(img)
    binary,_ = auto_threshold_otsu(gray)
    edges = detect_edges_canny(gray)
    lines = detect_lines_hough(edges)
    lines = filter_lines_by_length(lines, cfg.get("filter_min_length",50))
    h,w = gray.shape
    lines_img = draw_lines_on_blank((h,w), lines)
    combined = combine_line_images(lines_img)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(output_dir)/"combined.png"), combined)
    return combined

def main():
    p = argparse.ArgumentParser(description="Skizze – line art generator")
    p.add_argument("--input", required=True, help="Input image")
    p.add_argument("--output-dir", help="Where to save results")
    args = p.parse_args()
    process_image(args.input, args.output_dir)

if __name__ == "__main__":
    main()
