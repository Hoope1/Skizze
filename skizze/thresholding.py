import cv2
import numpy as np
from skimage.filters import threshold_local
from skimage.morphology import remove_small_objects, remove_small_holes


def auto_threshold_otsu(image_gray: np.ndarray) -> (np.ndarray, float):
    blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    retval, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, retval


def adaptive_threshold(image_gray: np.ndarray, block_size: int = 51, c: int = 10) -> np.ndarray:
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c
    )


def local_threshold_scikit(image_gray: np.ndarray, block_size: int = 51, offset: int = 10) -> np.ndarray:
    local_thresh = threshold_local(image_gray, block_size, offset=offset, method="gaussian")
    return (image_gray > local_thresh).astype(np.uint8) * 255


def multi_scale_binarization(image_gray: np.ndarray, scales=(1, 0.5)) -> np.ndarray:
    masks = []
    for scale in scales:
        if scale != 1:
            resized = cv2.resize(image_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            resized = image_gray
        bin_img, _ = auto_threshold_otsu(resized)
        if scale != 1:
            bin_img = cv2.resize(bin_img, (image_gray.shape[1], image_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        masks.append(bin_img)
    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)
    return combined


def clean_binary_scikit(binary: np.ndarray, min_size: int = 64) -> np.ndarray:
    bool_img = binary > 0
    clean = remove_small_objects(bool_img, min_size=min_size)
    clean = remove_small_holes(clean, area_threshold=min_size)
    return (clean.astype(np.uint8) * 255)


def adaptive_multi_threshold(image_gray: np.ndarray, n_clusters: int = 3) -> (np.ndarray, float):
    pixels = image_gray.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 10, flags)
    centers = np.sort(centers.flatten())
    thresh_val = centers[len(centers) // 2]
    _, binary = cv2.threshold(image_gray, thresh_val, 255, cv2.THRESH_BINARY)
    return binary, thresh_val
