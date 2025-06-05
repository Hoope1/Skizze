import cv2
import numpy as np
from .utils import download_model_if_missing


def detect_edges_canny(image_gray: np.ndarray, low_thresh: float = None, high_thresh: float = None) -> np.ndarray:
    if low_thresh is None or high_thresh is None:
        v = np.median(image_gray)
        sigma = 0.33
        low = int(max(0, (1.0 - sigma) * v))
        high = int(min(255, (1.0 + sigma) * v))
    else:
        low, high = low_thresh, high_thresh
    return cv2.Canny(image_gray, low, high, apertureSize=3)


def detect_lines_hough(
    edges: np.ndarray,
    rho: float = 1,
    theta: float = np.pi / 180,
    threshold: int = 100,
    min_line_len: int = 50,
    max_line_gap: int = 10,
) -> np.ndarray:
    lines = cv2.HoughLinesP(
        edges,
        rho,
        theta,
        threshold,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return np.empty((0, 4), dtype=np.int32)
    return lines.reshape(-1, 4)


def filter_lines_by_length(lines: np.ndarray, min_length: float = 50.0) -> np.ndarray:
    if lines.size == 0:
        return lines
    filtered = []
    for x1, y1, x2, y2 in lines:
        length = np.hypot(x2 - x1, y2 - y1)
        if length >= min_length:
            filtered.append([x1, y1, x2, y2])
    return np.array(filtered, dtype=np.int32)


# Deep Learning basierte Varianten -------------------------------------------------
try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - optional
    ort = None


def deep_lsd_line_detection(image_gray: np.ndarray, model_path: str) -> np.ndarray:
    if ort is None:
        raise RuntimeError("onnxruntime nicht verfügbar")
    download_model_if_missing(model_path, model_path)
    # Platzhalter: hier müsste eigentlich eine richtige Vorverarbeitung stehen
    session = ort.InferenceSession(model_path)
    input_data = image_gray.astype(np.float32)[None, None, :, :] / 255.0
    attr_field = session.run(None, {session.get_inputs()[0].name: input_data})[0]
    # Placeholder: simple thresholding on attraction field
    edges = (attr_field[0, 0] > 0.5).astype(np.uint8) * 255
    lines = detect_lines_hough(edges)
    return lines


try:
    import torch
    from fclip import FClipNetwork  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None
    FClipNetwork = None


def get_device():
    if torch is None:
        return None
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def fclip_line_detection(image_gray: np.ndarray, model_weights: str) -> np.ndarray:
    if FClipNetwork is None:
        raise RuntimeError("FClip nicht verfügbar")
    download_model_if_missing(model_weights, model_weights)
    device = get_device()
    net = FClipNetwork(weights=model_weights)
    if device is not None:
        net.to(device)
    tensor = torch.from_numpy(image_gray).unsqueeze(0).unsqueeze(0).float()
    if device is not None:
        tensor = tensor.to(device)
    detections = net(tensor)
    detections = detections.cpu().numpy()
    lines = detections[:, :4].astype(np.int32)
    return lines
