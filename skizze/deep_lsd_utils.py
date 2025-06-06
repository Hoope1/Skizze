import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def postprocess_attraction_field(attr_field: np.ndarray, orig_h:int, orig_w:int, conf_th:float=0.7):
    g0 = gaussian_filter(attr_field[0], sigma=1.0)
    g1 = gaussian_filter(attr_field[1], sigma=1.0)
    mag = np.sqrt(g0**2 + g1**2)
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    edges = (mag_norm > conf_th).astype(np.uint8)*255
    edges = cv2.Canny(edges, 50, 150)
    return cv2.resize(edges, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
