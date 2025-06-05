import cv2
import numpy as np


def draw_lines_on_blank(image_shape: tuple, lines: np.ndarray, color: tuple = (255, 255, 255), thickness: int = 1) -> np.ndarray:
    blank = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    for x1, y1, x2, y2 in lines:
        cv2.line(blank, (x1, y1), (x2, y2), color, thickness)
    return blank


def draw_contours_on_blank(image_shape: tuple, contours: list, color: tuple = (255, 255, 255), thickness: int = 1) -> np.ndarray:
    blank = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        for i in range(len(pts) - 1):
            cv2.line(blank, tuple(pts[i]), tuple(pts[i + 1]), color, thickness)
    return blank


def combine_line_images(*images: np.ndarray) -> np.ndarray:
    if len(images) == 0:
        raise ValueError("Mindestens ein Bild muss übergeben werden.")
    h, w = images[0].shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    for img in images:
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bin_mask = (gray > 0).astype(np.uint8) * 255
        combined = cv2.bitwise_or(combined, bin_mask)
    return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
