import numpy as np
try:
    from skimage.morphology import remove_small_objects, remove_small_holes, skeletonize
    _SK_IMG = True
except ImportError:
    _SK_IMG = False

def skeletonize_image(image_bin: np.ndarray):
    if not _SK_IMG:
        print("⚠️ skimage not available: skipping skeletonize.")
        return image_bin
    skel = skeletonize((image_bin>0))
    return (skel.astype(np.uint8)*255)

def clean_binary_scikit(binary_img: np.ndarray, min_size:int=64):
    if not _SK_IMG:
        raise ImportError("scikit-image not installed.")
    bool_img = binary_img>0
    cleaned = remove_small_objects(bool_img, min_size=min_size)
    cleaned = remove_small_holes(cleaned, area_threshold=min_size)
    return (cleaned.astype(np.uint8)*255)
