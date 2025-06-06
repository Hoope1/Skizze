import cv2
import numpy as np

def draw_lines_on_blank(image_shape, lines, color=(255,255,255), thickness=1):
    h,w = image_shape
    blank = np.zeros((h,w,3), np.uint8)
    for x1,y1,x2,y2 in lines:
        cv2.line(blank,(int(x1),int(y1)),(int(x2),int(y2)),color,thickness)
    return blank

def draw_contours_on_blank(image_shape, contours, color=(255,255,255), thickness=1):
    h,w = image_shape
    blank = np.zeros((h,w,3), np.uint8)
    for cnt in contours:
        pts = cnt.reshape(-1,2)
        for i in range(len(pts)-1):
            x0,y0 = map(int, pts[i])
            x1,y1 = map(int, pts[i+1])
            cv2.line(blank,(x0,y0),(x1,y1),color,thickness)
    return blank

def combine_line_images(*images):
    if not images:
        raise ValueError("At least one image required.")
    h,w = images[0].shape[:2]
    combined = np.zeros((h,w), np.uint8)
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        mask = (gray>0).astype(np.uint8)*255
        combined = cv2.bitwise_or(combined, mask)
    return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
