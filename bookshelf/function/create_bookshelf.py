# https://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/
# using nothing but the mean and standard flashwomaniation of the image channels (Lab)

# pip install color_transfer
from color_transfer import color_transfer
import cv2 as cv
import numpy as np
from bookshelf.function.display import display


def create_bookshelf(img_color):

    # load the images
    path = 'C:/flashwoman/Object-detection/image/emptybookshelf_box.jpg'

    # color_transfer
    source = img_color
    target = cv.imread(path)
    transfer = color_transfer(source, target)

    # get height, width
    height, width = target.shape[:2]
    print(width,height)

    # box 쌓기
    boxes = []
    coords = []
    for i in range(0, 4 ):
        boxes.append(transfer)

    res = np.vstack(boxes)
    path = "C:/flashwoman/Object-detection/image/bookshelf_fin.jpg"
    cv.imwrite(path, res)
    boxes = cv.imread("C:/flashwoman/Object-detection/image/bookshelf_fin.jpg")

    # display("Source", source)
    # display("Target", target)
    # display("Transfer", transfer)
    # display("boxes", boxes)

    for i in range(0, 4):
        h_w = height * 1.01 * i
        x1 = width * 0.0157
        x2 = width - x1
        y = height * 0.93 + h_w

        x1,x2, y = list(map(int, [x1,x2, y]))

        coord_1 = (x1, y)
        coord_2 = (x2, y)
        coords.append(  [coord_1,coord_2]  )
        cv.circle(boxes, coord_1, 2, (0, 0, 255), -1)
        cv.circle(boxes, coord_2, 2, (0, 0, 255), -1)

    print(coords)
    display("boxes2", boxes)
    cv.waitKey(0)

    return width, height, coords