import cv2 as cv


def img_write(file_name, img_color):

    path = f"C:/Users/DELL/PycharmProjects/EyeCandy/img/result/{file_name}.jpg"
    cv.imwrite(path, img_color)