import cv2
import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import interpolation as inter

def enlarge_img(image: np.ndarray, target_dim=(700, 280)) -> np.ndarray:
    """Resizes the license plate ROI image for optimal OCR extraction."""
    if image is None or image.size == 0:
        return image
    return cv2.resize(image, target_dim, interpolation=cv2.INTER_CUBIC)

def find_score(arr: np.ndarray, angle: float):
    """Calculates projection profile variance score for a given rotation angle."""
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def set_angle(gray_img: np.ndarray) -> np.ndarray:
    """Deskews license plate image by computing vertical projection profile scores across angles."""
    if len(gray_img.shape) == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        
    pil_img = PILImage.fromarray(gray_img)
    wd, ht = pil_img.size
    pix = np.array(pil_img.convert('1').getdata(), np.uint8)
    bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)

    delta = 1
    limit = 7
    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    
    for angle in angles:
        _, score = find_score(bin_img, angle)
        scores.append(score)
        
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]

    rotated_data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
    rotated_pil = PILImage.fromarray((255 * rotated_data).astype("uint8")).convert("RGB")
    deskewed_cv = cv2.cvtColor(np.array(rotated_pil), cv2.COLOR_RGB2GRAY)
    return deskewed_cv

def dilate(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def erode(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def Preprocessing(img: np.ndarray) -> np.ndarray:
    """
    Applies comprehensive image enhancement pipeline:
    1. Resizing to target resolution
    2. Grayscale conversion & angle deskewing
    3. Gaussian blurring & Otsu binary thresholding
    4. Bilateral filtering
    5. Morphological Dilation and Erosion
    """
    if img is None or img.size == 0:
        return img
        
    resized = enlarge_img(img)
    
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized.copy()

    deskewed = set_angle(gray)

    blur = cv2.GaussianBlur(deskewed, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    filtered = cv2.bilateralFilter(thresh, 11, 17, 17)

    dilated = dilate(filtered)
    eroded = erode(dilated)

    return eroded
