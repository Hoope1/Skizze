import cv2
import numpy as np
import logging
from .utils import ensure_directory

logger = logging.getLogger(__name__)

try:
    from skimage.morphology import skeletonize
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Bild unter '{path}' konnte nicht geladen werden.")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise_image(image_gray: np.ndarray, method: str = "bilateral") -> np.ndarray:
    if method == "bilateral":
        return cv2.bilateralFilter(image_gray, d=9, sigmaColor=75, sigmaSpace=75)
    if method == "median":
        return cv2.medianBlur(image_gray, ksize=5)
    return image_gray


def morphological_opening(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def morphological_closing(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def skeletonize_image(image_bin: np.ndarray) -> np.ndarray:
    if not SKIMAGE_AVAILABLE:
        logger.warning("skimage nicht verfügbar. Rückgabe des Originalbilds.")
        return image_bin
    bin01 = (image_bin > 0).astype(np.uint8)
    skeleton = skeletonize(bin01)
    return (skeleton * 255).astype(np.uint8)
