import cv2 as cv
import os
import glob
from .display import display


def contours(img_book_only, img_color):

    img_for_save = img_color.copy()

    # 1. Threshold
    ret, img_binary = cv.threshold(img_book_only, 127, 255, 0)
    # display("img_binary", img_binary)

    # 2. Contour 찾기
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # 3. 기존에 저장된 이미지 파일 삭제
    path = "C:/flashwoman/Object-detection/image/result/"
    files = '*.jpg'  # 찾고 싶은 확장자
    for file in glob.glob(os.path.join(path, files)):
        os.remove(file)

    ## 4. countour 그리고 저장하기.
    count = 0
    rect = []
    for cnt in contours:
        ## 1.  사각형 만들어 좌표얻기.
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(img_color, (x, y), (x + w, y + h), (3, 255, 4), 2)  # contour 그리기
        rect.append([(x, y), (x + w, y + h)])  # rect에 좌표 저장 [Left_Top, Right_Bottom]

        ## 2. 각각의 책 이미지 저장 ##
        img_result = img_for_save[y: y + h, x: x + w]
        img_path = f"C:/flashwoman/Object-detection/image/result/book/img{count}.jpg"
        cv.imwrite(img_path, img_result)

        count = count + 1

    path = "C:/flashwoman/Object-detection/image/result/result.jpg"
    # display("result", img_color)
    cv.imwrite(path, img_color)


    return img_book_only, rect