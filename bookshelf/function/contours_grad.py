import cv2 as cv
import os
import glob
import numpy as np
from bookshelf.function.display import display


def contours_grad(img_book_only, img_color):

    img_for_save = img_color.copy()

    # 1. Threshold
    ret, img_binary = cv.threshold(img_book_only, 127, 255, 0)

    # 2. Contour 찾기
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # 3. 기존에 저장된 이미지 파일 삭제
    path = "C:/flashwoman/Object-detection/EyeCandy/books/"
    files = '*.jpg'  # 찾고 싶은 확장자
    for file in glob.glob(os.path.join(path, files)):
        os.remove(file)

    # 4. contour 그리기 + 저장하기
    boxes = []
    moments = []
    rects = []

    for cnt in contours:
        # Rotated Rectangle
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        M = cv.moments(cnt)

        rects.append(rect)
        boxes.append(box.astype('int'))
        moments.append(M["m00"])

    mean = np.mean(moments)
    median = np.median(moments)
    rect = []
    for i, cnt in enumerate(contours):
        if (hierarchy[0][i][3] != -1) & (moments[i] > mean) & (moments[i] < mean * 2.5):
            cv.drawContours(img_color, [boxes[i]], -1, (3, 255, 4), 1)  # blue

            (x, y) = rects[i][0]
            (w, h) = rects[i][1]

            x, y, w, h = list(map(int, [x, y, w, h]))

            ## 2. 각각의 책 이미지 저장 ##
            img_result = img_for_save[y: y + h, x: x + w]
            img_path = f"{path}/img{i}.jpg"
            cv.imwrite(img_path, img_result)

    # display('img_color', img_color)
    return img_book_only, rect