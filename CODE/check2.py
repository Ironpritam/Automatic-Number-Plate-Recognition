import cv2
#import pytesseract
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter
#import copy


def Preprocessing(img):
    img = enlarge_img(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = set_angle(img)

    blur = cv2.GaussianBlur(img, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise and invert image
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    #invert = 255 - opening


    #img = cv2.bitwise_not(img)
    img = cv2.bilateralFilter(thresh, 11, 17, 17)

    #img = cv2.adaptiveThreshold(cv2.bilateralFilter(img,9,75,75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,25,1)
    img = dilate(img)
    img = erode(img)
    #img = canny(img)
    #img = set_angle(img)
    #img = cv2.adaptiveThreshold(cv2.bilateralFilter(img,9,75,75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,25,1)
    #img = cv2.GaussianBlur(img, (5, 5), 0)

    #img = erode(img)

    #img = cv2.bitwise_not(img)


    #img = remove_noise_and_smooth(img)
    #img = combine(img)

    #cv2.imshow("test",img)
    #cv2.waitKey(0)
    return img

def enlarge_img(image, scale_percent=400):
    #width = int(image.shape[1] * scale_percent / 100)
    #height = int(image.shape[0] * scale_percent / 100)
    dim = (700,280 )
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
    return resized_image


def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def set_angle(img):
    img = im.fromarray(img)
    wd, ht = img.size
    pix = np.array(img.convert('1').getdata(), np.uint8)
    bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)

    delta = 1
    limit = 7
    angles = np.arange(-limit, limit+delta, delta)
    #print(angles)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
#   print('Best angle: {}'.formate(best_angle))
#   correct skew
    data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
    img = im.fromarray((255 * data).astype("uint8")).convert("RGB")
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return(opencvImage)

def combine(image):
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("1 - Grayscale Conversion", gray)

    gray = cv2.bilateralFilter(image, 11, 17, 17)
    #cv2.imshow("2 - Bilateral Filter", gray)

    edged = cv2.Canny(gray, 170, 200)
    #cv2.imshow("4 - Canny Edges", edged)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
    NumberPlateCnt = None

    count = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            break

    # Masking the part other than the number plate
    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
    new_image = cv2.bitwise_and(image,image,mask=mask)
    return(new_image)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((3,3),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

#erosion
def erode(image):
    kernel = np.ones((3,3),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((3,3),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


def remove_noise_and_smooth(img):
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = cv2.medianBlur(img, 3)
    or_image = cv2.bitwise_or(img, closing)
    #th, or_image = cv2.threshold(or_image, 128, 192, cv2.THRESH_BINARY)
    #or_image = cv2.bitwise_not(or_image)

    return or_image
