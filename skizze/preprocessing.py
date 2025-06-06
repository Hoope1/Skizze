import cv2
import numpy as np

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image '{path}'.")
    return img

def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def denoise_image(image_gray: np.ndarray, method: str = None) -> np.ndarray:
    if method == "bilateral":
        return cv2.bilateralFilter(image_gray, 9, 75, 75)
    elif method == "median":
        return cv2.medianBlur(image_gray, 5)
    return image_gray

def morphological_opening(binary_img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, k)

def morphological_closing(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, k)
