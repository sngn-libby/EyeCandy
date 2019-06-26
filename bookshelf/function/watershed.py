import cv2 as cv
import numpy as np
from .display import display


def watershed(img_color, m=3, n=3):

    img_blur = cv.medianBlur(img_color, 5)
    img_gray = cv.cvtColor(img_blur, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    # noise removal(optional)
    if m != 3:
        if n != 3:
            kernel = np.ones((m, n), np.uint8)
        else:
            kernel = np.ones((m, 3), np.uint8)
    else:
        kernel = np.ones((3, 3), np.uint8)

    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv.dilate(opening, kernel, iterations=3)
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255,0)

    unknown = cv.subtract(sure_bg, sure_fg)
    # display(unknown)

    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers +1
    markers[unknown==255] = 0

    # watershed
    img_watershed = cv.watershed(img_color, markers)

    # display('thresh_1', thresh)
    # display('dist_transform', dist_transform)
    # display('sure_foreground', sure_fg)
    # display('markers', markers)
    # display('watershed result', img_watershed)

    return img_watershed

# img = cv.imread('C:/fleshwoman/Object-detection/image/water_coins.jpg')
# img[markers == -1] = [255,0,0]
