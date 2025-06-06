import cv2
import numpy as np
import torch
from .utils import download_model_if_missing
from .deep_lsd_utils import postprocess_attraction_field

def detect_edges_canny(image_gray, low_thresh=None, high_thresh=None):
    if low_thresh is None or high_thresh is None:
        v = np.median(image_gray)
        sigma = 0.33
        low_thresh = int(max(0, (1.0-sigma)*v))
        high_thresh = int(min(255,(1.0+sigma)*v))
    return cv2.Canny(image_gray, low_thresh, high_thresh, apertureSize=3)

def detect_lines_hough(edges, rho=1.0, theta=np.pi/180, threshold=100, min_line_len=50, max_line_gap=10):
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None:
        return np.empty((0,4), np.int32)
    return lines.reshape(-1,4)

def filter_lines_by_length(lines, min_length=50.0):
    if lines.size==0:
        return lines
    out=[]
    for x1,y1,x2,y2 in lines:
        if np.hypot(x2-x1, y2-y1) >= min_length:
            out.append([x1,y1,x2,y2])
    return np.array(out, np.int32)

def deep_lsd_line_detection(image_gray, model_path, conf_th=0.7):
    download_model_if_missing(model_path, model_path)
    import onnxruntime as ort
    img = image_gray.astype(np.float32)/255.
    h,w = img.shape
    pad_h = (32 - h%32)%32
    pad_w = (32 - w%32)%32
    img_pad = np.pad(img, ((0,pad_h),(0,pad_w)), mode='constant')
    inp = img_pad[None,None,:,:]
    session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    attr = session.run(None, {session.get_inputs()[0].name: inp})[0][0]
    edges = postprocess_attraction_field(attr, h, w, conf_th)
    return detect_lines_hough(edges)

def fclip_line_detection(image_gray, model_weights, device):
    from .fclip_net import FClipNetwork
    net = FClipNetwork().to(device)
    chk = torch.load(model_weights, map_location=device)
    if "state_dict" in chk:
        net.load_state_dict(chk["state_dict"])
    else:
        net.load_state_dict(chk)
    net.eval()
    img = image_gray.astype(np.float32)/255.
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        det = net(tensor)
    if isinstance(det, torch.Tensor):
        det_np = det.cpu().numpy().astype(np.int32)
    else:
        det_np = np.array(det, np.int32)
    return det_np
