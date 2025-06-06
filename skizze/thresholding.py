import cv2
import numpy as np
try:
    from skimage.filters import threshold_local
    _SK_AVAILABLE = True
except ImportError:
    _SK_AVAILABLE = False

def auto_threshold_otsu(image_gray: np.ndarray):
    blur = cv2.GaussianBlur(image_gray, (5,5), 0)
    val, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, float(val)

def adaptive_threshold(image_gray: np.ndarray, block_size: int = 51, c: int = 10):
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(image_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,block_size,c)

def local_threshold_scikit(image_gray: np.ndarray, block_size:int=51, offset:int=10, method:str="gaussian"):
    if not _SK_AVAILABLE:
        raise ImportError("scikit-image not installed.")
    thresh = threshold_local(image_gray, block_size, offset=offset, method=method)
    return (image_gray > thresh).astype(np.uint8)*255

def multi_scale_threshold(image_gray: np.ndarray, base_block:int=51, offset:int=10, levels:int=3):
    h,w = image_gray.shape
    combined = np.zeros((h,w), np.uint8)
    curr = image_gray.copy()
    for lvl in range(levels):
        blk = max(3, base_block // (2**lvl))
        if blk % 2 == 0:
            blk += 1
        mask = cv2.adaptiveThreshold(curr,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blk,offset)
        if lvl>0:
            mask = cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
        combined = cv2.bitwise_or(combined, mask)
        if lvl<levels-1:
            curr = cv2.pyrDown(curr)
    return combined

def adaptive_multi_threshold(image_gray: np.ndarray, n_clusters:int=3):
    pixels = image_gray.reshape(-1,1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.2)
    ret,labels,centers = cv2.kmeans(pixels,n_clusters,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    centers = np.sort(centers.flatten())
    thresh = float(centers[len(centers)//2])
    _, binary = cv2.threshold(image_gray, thresh, 255, cv2.THRESH_BINARY)
    return binary, thresh
