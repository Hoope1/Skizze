from pathlib import Path
import argparse
import logging

from .utils import ensure_venv, install_missing_packages, REQUIRED_PACKAGES, setup_logging

# configure environment
setup_logging()
ensure_venv()
install_missing_packages(REQUIRED_PACKAGES)

import numpy as np
import cv2
import torch

from .config import load_config_from_pyproject
from .preprocessing import (
    load_image,
    to_grayscale,
    denoise_image,
    morphological_opening,
    morphological_closing,
)
from .thresholding import (
    auto_threshold_otsu,
    adaptive_threshold,
    local_threshold_scikit,
    multi_scale_threshold,
    adaptive_multi_threshold,
)
from .drawing import draw_lines_on_blank, draw_contours_on_blank, combine_line_images
from .line_detection import (
    detect_edges_canny,
    detect_lines_hough,
    filter_lines_by_length,
    deep_lsd_line_detection,
    fclip_line_detection,
)
from .scikit_tools import clean_binary_scikit, skeletonize_image

log = logging.getLogger(__name__)
cfg = load_config_from_pyproject()


def save_image(img: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def process_image(image_path: str, output_dir: Path, args) -> None:
    log.info("Processing %s", image_path)
    img = load_image(image_path)
    gray = to_grayscale(img)
    gray = denoise_image(gray, args.denoise_method or cfg.get("denoise_method", ""))

    if args.no_otsu:
        binary = gray
    else:
        binary, _ = auto_threshold_otsu(gray)

    if args.use_scikit_local:
        try:
            binary = local_threshold_scikit(
                binary,
                block_size=args.scikit_local_block,
                offset=args.scikit_local_offset,
            )
        except ImportError:
            log.warning("scikit-image required for local threshold. Skipping.")
    elif args.use_multi_scale:
        binary = multi_scale_threshold(
            binary,
            base_block=args.ms_block,
            offset=args.ms_offset,
            levels=args.ms_levels,
        )
    elif args.use_kmeans_thresh:
        binary, _ = adaptive_multi_threshold(binary, args.kmeans_clusters)
    else:
        binary = adaptive_threshold(binary, block_size=args.adaptive_block, c=args.adaptive_c)

    binary = morphological_opening(binary)
    if args.clean_small_objects:
        try:
            binary = clean_binary_scikit(binary, args.min_small_size)
        except ImportError:
            log.warning("scikit-image required for cleaning. Skipping.")
    binary = morphological_closing(binary)

    edges = detect_edges_canny(gray)

    if args.use_hough_only:
        lines = detect_lines_hough(edges)
    elif args.use_deep_lsd:
        lines = deep_lsd_line_detection(gray, args.deep_lsd_model, args.conf_th)
    elif args.use_fclip:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        lines = fclip_line_detection(gray, args.fclip_model, device)
    else:
        lines = detect_lines_hough(edges)

    lines = filter_lines_by_length(lines, args.min_line_len)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    skel = skeletonize_image(binary) if args.do_skeleton else None

    h, w = gray.shape
    lines_img = draw_lines_on_blank((h, w), lines)
    contours_img = draw_contours_on_blank((h, w), contours)
    imgs = [lines_img, contours_img]
    if skel is not None:
        skel_img = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        imgs.append(skel_img)
    combined = combine_line_images(*imgs)

    if output_dir:
        out = Path(output_dir)
        save_image(gray, out / "grayscale.png")
        save_image(binary, out / "binary.png")
        save_image(edges, out / "edges.png")
        save_image(lines_img, out / "lines.png")
        save_image(contours_img, out / "contours.png")
        if skel is not None:
            save_image(skel, out / "skeleton.png")
        save_image(combined, out / "combined.png")


def process_batch(paths, output_root: Path, args) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    def worker(p):
        out_dir = output_root / Path(p).stem
        process_image(p, out_dir, args)
        return p

    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(worker, p) for p in paths]
        for f in as_completed(futures):
            log.info("Finished %s", f.result())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Skizze – line art generator")
    p.add_argument("--input", help="Input image")
    p.add_argument("--output-dir", default="results", help="Where to save results")

    # thresholding
    p.add_argument("--no-otsu", action="store_true", help="Skip Otsu threshold")
    p.add_argument("--adaptive-block", type=int, default=cfg.get("adaptive_block", 51))
    p.add_argument("--adaptive-c", type=int, default=cfg.get("adaptive_c", 10))
    p.add_argument("--use-scikit-local", action="store_true")
    p.add_argument("--scikit-local-block", type=int, default=cfg.get("adaptive_block", 51))
    p.add_argument("--scikit-local-offset", type=int, default=cfg.get("adaptive_c", 10))
    p.add_argument("--use-multi-scale", action="store_true")
    p.add_argument("--ms-block", type=int, default=cfg.get("ms_block", 51))
    p.add_argument("--ms-offset", type=int, default=cfg.get("ms_offset", 10))
    p.add_argument("--ms-levels", type=int, default=cfg.get("ms_levels", 3))
    p.add_argument("--use-kmeans-thresh", action="store_true")
    p.add_argument("--kmeans-clusters", type=int, default=3)

    # morphology
    p.add_argument("--clean-small-objects", action="store_true")
    p.add_argument("--min-small-size", type=int, default=cfg.get("min_small_size", 64))

    # denoising
    p.add_argument("--denoise-method", choices=["bilateral", "median"], default=cfg.get("denoise_method", ""))

    # line detection
    p.add_argument("--use-hough-only", action="store_true")
    p.add_argument("--use-deep-lsd", action="store_true")
    p.add_argument("--deep-lsd-model")
    p.add_argument("--conf-th", type=float, default=cfg.get("conf_th", 0.7))
    p.add_argument("--use-fclip", action="store_true")
    p.add_argument("--fclip-model")
    p.add_argument("--device", default=cfg.get("device", ""))
    p.add_argument("--min-line-len", type=int, default=int(cfg.get("filter_min_length", 50)))

    # skeletonization
    p.add_argument("--do-skeleton", action="store_true")

    # batch
    p.add_argument("--batch-dir")
    p.add_argument("--batch-list")

    # logging
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--log-file")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level, args.log_file)

    if args.batch_dir or args.batch_list:
        if args.batch_list:
            with open(args.batch_list) as f:
                paths = [line.strip() for line in f if line.strip()]
        else:
            extensions = {".jpg", ".jpeg", ".png", ".bmp"}
            paths = [str(p) for p in Path(args.batch_dir).iterdir() if p.suffix.lower() in extensions]
        process_batch(paths, Path(args.output_dir), args)
    else:
        if not args.input:
            parser.error("--input is required unless batch mode is used")
        process_image(args.input, Path(args.output_dir), args)


if __name__ == "__main__":
    main()
