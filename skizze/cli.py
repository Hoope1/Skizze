import os
import sys
import subprocess
from pathlib import Path
import venv


def ensure_venv():
    project_root = Path(__file__).parent.resolve()
    venv_dir = project_root / "env"

    if str(sys.prefix).startswith(str(venv_dir)):
        return

    if not venv_dir.exists():
        print("\ud83d\udd27 Erstelle virtuelle Umgebung in './env' …")
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(str(venv_dir))

    python_exe = venv_dir / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )

    new_argv = [str(python_exe)] + sys.argv
    os.execv(str(python_exe), new_argv)


ensure_venv()


REQUIRED_PACKAGES = [
    "numpy",
    "opencv-python",
    "scikit-image",
    "onnxruntime",
    "torch",
    "requests",
    "toml",
]


def install_missing_packages(packages):
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"\ud83d\udce6 Paket '{pkg}' fehlt. Installiere es jetzt …")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


install_missing_packages(REQUIRED_PACKAGES)

import cv2
import argparse
import tomllib

from .preprocessing import (
    load_image,
    to_grayscale,
    denoise_image,
    morphological_opening,
    morphological_closing,
    skeletonize_image,
)
from .thresholding import (
    auto_threshold_otsu,
    adaptive_threshold,
    local_threshold_scikit,
    multi_scale_binarization,
    adaptive_multi_threshold,
    clean_binary_scikit,
)
from .line_detection import (
    detect_edges_canny,
    detect_lines_hough,
    filter_lines_by_length,
    deep_lsd_line_detection,
    fclip_line_detection,
)
from .drawing import (
    draw_lines_on_blank,
    draw_contours_on_blank,
    combine_line_images,
)
from .utils import (
    ensure_directory,
    extract_contours,
)


DEFAULT_CONFIG = {
    "adaptive_block": 51,
    "adaptive_c": 10,
    "min_line_len": 50.0,
    "hough_threshold": 100,
    "ms_block": 51,
    "ms_offset": 10,
    "ms_levels": 3,
    "scikit_block": 51,
    "scikit_offset": 10,
    "kmeans_clusters": 3,
}

CONFIG_PATH = Path("pyproject.toml")
if CONFIG_PATH.exists():
    data = tomllib.loads(CONFIG_PATH.read_text())
    DEFAULT_CONFIG.update(data.get("tool", {}).get("skizze", {}))


def process_image(
    path: str,
    output_dir: str | None = None,
    use_otsu: bool = True,
    adaptive_block: int = 51,
    adaptive_c: int = 10,
    use_scikit_local: bool = False,
    scikit_block: int = 51,
    scikit_offset: int = 10,
    use_multi_scale: bool = False,
    ms_block: int = 51,
    ms_offset: int = 10,
    ms_levels: int = 3,
    use_kmeans: bool = False,
    kmeans_clusters: int = 3,
    use_deep_lsd: bool = False,
    deep_lsd_model: str | None = None,
    use_fclip: bool = False,
    fclip_model: str | None = None,
    denoise_method: str | None = None,
    filter_min_length: float = 50.0,
    do_skeleton: bool = False,
):
    img_color = load_image(path)
    gray = to_grayscale(img_color)
    if denoise_method:
        gray = denoise_image(gray, method=denoise_method)
    else:
        gray = denoise_image(gray)

    if use_scikit_local:
        bin_img = local_threshold_scikit(gray, block_size=scikit_block, offset=scikit_offset)
        thresh_val = None
    elif use_multi_scale:
        bin_img = multi_scale_binarization(gray)
        thresh_val = None
    elif use_kmeans:
        bin_img, thresh_val = adaptive_multi_threshold(gray, n_clusters=kmeans_clusters)
    elif use_otsu:
        bin_img, thresh_val = auto_threshold_otsu(gray)
    else:
        bin_img = adaptive_threshold(gray, block_size=adaptive_block, c=adaptive_c)
        thresh_val = None

    bin_img = morphological_opening(bin_img)
    bin_img = clean_binary_scikit(bin_img)

    edges = detect_edges_canny(gray)
    if use_deep_lsd:
        if not deep_lsd_model:
            raise ValueError("--deep-lsd-model muss angegeben werden")
        lines = deep_lsd_line_detection(gray, deep_lsd_model)
    elif use_fclip:
        if not fclip_model:
            raise ValueError("--fclip-model muss angegeben werden")
        lines = fclip_line_detection(gray, fclip_model)
    else:
        lines = detect_lines_hough(edges, threshold=DEFAULT_CONFIG.get("hough_threshold", 100))
        lines = filter_lines_by_length(lines, min_length=filter_min_length)

    contours = extract_contours(bin_img)

    if do_skeleton:
        skeleton = skeletonize_image(bin_img)
    else:
        skeleton = None

    h, w = gray.shape
    lines_img = draw_lines_on_blank((h, w), lines)
    contours_img = draw_contours_on_blank((h, w), contours)

    if skeleton is not None:
        skeleton_bgr = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        combined = combine_line_images(lines_img, contours_img, skeleton_bgr)
    else:
        combined = combine_line_images(lines_img, contours_img)

    combined = morphological_closing(combined)
    results = {
        "binary": bin_img,
        "edges": edges,
        "lines": lines,
        "contours": contours,
        "combined": combined,
        "threshold_value": thresh_val,
    }

    if output_dir:
        ensure_directory(output_dir)
        cv2.imwrite(str(Path(output_dir) / "binary.png"), bin_img)
        cv2.imwrite(str(Path(output_dir) / "edges.png"), edges)
        cv2.imwrite(str(Path(output_dir) / "lines.png"), lines_img)
        cv2.imwrite(str(Path(output_dir) / "contours.png"), contours_img)
        cv2.imwrite(str(Path(output_dir) / "combined.png"), combined)

    return results

def process_batch(paths, output_dir, args):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_image,
                p,
                output_dir,
                not args.no_otsu,
                args.adaptive_block,
                args.adaptive_c,
                args.use_scikit_local,
                args.scikit_block,
                args.scikit_offset,
                args.use_multi_scale,
                args.ms_block,
                args.ms_offset,
                args.ms_levels,
                args.use_kmeans,
                args.kmeans_clusters,
                args.use_deep_lsd,
                args.deep_lsd_model,
                args.use_fclip,
                args.fclip_model,
                args.denoise_method,
                args.min_line_len,
                args.do_skeleton,
            )
            for p in paths
        ]
        for f in futures:
            f.result()


def main():
    parser = argparse.ArgumentParser(description="Optimale Linien zum Abpausen aus einem Bild extrahieren.")
    parser.add_argument("--input", required=True, help="Pfad zum Eingabebild.")
    parser.add_argument("--batch", nargs="*", help="Liste weiterer Bilder zur Stapelverarbeitung.")
    parser.add_argument("--output-dir", default=None, help="Verzeichnis für Zwischenergebnisse.")
    parser.add_argument("--no-otsu", action="store_true", help="Deaktiviere Otsu.")
    parser.add_argument("--adaptive-block", type=int, default=DEFAULT_CONFIG["adaptive_block"], help="Blockgröße.")
    parser.add_argument("--adaptive-c", type=int, default=DEFAULT_CONFIG["adaptive_c"], help="C für adaptive Binarisierung.")
    parser.add_argument("--use-scikit-local", action="store_true", help="Lokales Thresholding mit scikit-image nutzen.")
    parser.add_argument("--scikit-block", type=int, default=DEFAULT_CONFIG["scikit_block"], help="Blockgröße für scikit-image local threshold.")
    parser.add_argument("--scikit-offset", type=int, default=DEFAULT_CONFIG["scikit_offset"], help="Offset für scikit-image local threshold.")
    parser.add_argument("--use-multi-scale", action="store_true", help="Multi-Scale-Binarisierung verwenden.")
    parser.add_argument("--ms-block", type=int, default=DEFAULT_CONFIG["ms_block"], help="Basisblock für Multi-Scale.")
    parser.add_argument("--ms-offset", type=int, default=DEFAULT_CONFIG["ms_offset"], help="Offset für Multi-Scale.")
    parser.add_argument("--ms-levels", type=int, default=DEFAULT_CONFIG["ms_levels"], help="Anzahl Ebenen für Multi-Scale.")
    parser.add_argument("--use-kmeans", action="store_true", help="K-Means-Threshold verwenden.")
    parser.add_argument("--kmeans-clusters", type=int, default=DEFAULT_CONFIG["kmeans_clusters"], help="Clusterzahl für K-Means.")
    parser.add_argument("--use-deep-lsd", action="store_true", help="DeepLSD zur Linienerkennung nutzen.")
    parser.add_argument("--deep-lsd-model", type=str, default=None, help="Pfad zum DeepLSD-ONNX-Modell.")
    parser.add_argument("--use-fclip", action="store_true", help="FClip zur Linienerkennung nutzen.")
    parser.add_argument("--fclip-model", type=str, default=None, help="Pfad zu den FClip-Gewichten.")
    parser.add_argument("--denoise-method", type=str, default=None, choices=["bilateral", "median"], help="Methode zur Rauschreduktion.")
    parser.add_argument("--min-line-len", type=float, default=DEFAULT_CONFIG["min_line_len"], help="Minimale Länge für Linien.")
    parser.add_argument("--do-skeleton", action="store_true", help="Skelettierung durchführen.")
    args = parser.parse_args()

    process_image(
        path=args.input,
        output_dir=args.output_dir,
        use_otsu=not args.no_otsu,
        adaptive_block=args.adaptive_block,
        adaptive_c=args.adaptive_c,
        use_scikit_local=args.use_scikit_local,
        scikit_block=args.scikit_block,
        scikit_offset=args.scikit_offset,
        use_multi_scale=args.use_multi_scale,
        ms_block=args.ms_block,
        ms_offset=args.ms_offset,
        ms_levels=args.ms_levels,
        use_kmeans=args.use_kmeans,
        kmeans_clusters=args.kmeans_clusters,
        use_deep_lsd=args.use_deep_lsd,
        deep_lsd_model=args.deep_lsd_model,
        use_fclip=args.use_fclip,
        fclip_model=args.fclip_model,
        denoise_method=args.denoise_method,
        filter_min_length=args.min_line_len,
        do_skeleton=args.do_skeleton,
    )
    if args.batch:
        process_batch(args.batch, args.output_dir, args)

if __name__ == "__main__":
    main()
