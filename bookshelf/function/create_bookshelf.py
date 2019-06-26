# https://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/
# using nothing but the mean and standard flashwomaniation of the image channels (Lab)

# pip install color_transfer
from color_transfer import color_transfer
import cv2 as cv
import numpy as np
# from bookshelf.function.display import display
# from bookshelf.function.img_write import img_write


def create_bookshelf(img_color, max_book_height):

    # load the images
    path = 'C:/flashwoman/Object-detection/image/emptybookshelf_box.jpg'

    ######################################## need to change target
    # color_transfer
    source = img_color
    target = cv.imread(path)
    height, width = target.shape[:2]
    prop_y = max_book_height / height
    target = cv.resize(target, (0, 0), None, fx=prop_y, fy=prop_y*0.8, interpolation=cv.INTER_AREA)
    # fy_prop = shelf_height / (target.shape[:2][1])
    # print("fy", fy_prop)
    # target = cv.resize(target, None, fy=fy_prop, interpolation=cv.INTER_AREA)
    transfer = color_transfer(source, target)

    # get height, width
    height, width = target.shape[:2]
    # print('bookshelf width/height :',width,height)

    # box 쌓기
    boxes = []
    coords_bottom = []
    coords_top = []

    for i in range(4):
        boxes.append(transfer)

    res = np.vstack(boxes)
    path = "C:/flashwoman/Object-detection/image/bookshelf_fin.jpg"
    cv.imwrite(path, res)
    boxes = cv.imread("C:/flashwoman/Object-detection/image/bookshelf_fin.jpg")

    # display("Source", source)
    # display("Target", target)
    # display("Transfer", transfer)
    # display("boxes", boxes)

    for i in range(4):

        h_width = height * 1.01
        x1 = width * 0.0157
        x2 = width - x1
        y1 = (height*0.04) + (height*i) # (h_width*i) # fit perfectly
        y2 = (height*0.96) + (height*i) # (h_width*i)

        x1,x2, y1,y2 = list(map(int, [x1,x2, y1,y2]))

        coord_1 = (x1, y1) # lb
        coord_2 = (x2, y1) # rb
        coord_3 = (x1, y2) # lt
        coord_4 = (x2, y2) # rt
        coords_bottom.append(  [coord_1,coord_2]  )
        coords_top.append(  [coord_3,coord_4]  )
        # cv.circle(boxes, coord_1, 2, (0, 0, 255), -1)
        # cv.circle(boxes, coord_2, 2, (0, 0, 255), -1)

        # # ex_boxes = boxes.copy() # 원본에 영향미치고 싶지 않다면 unhash
        # roi = boxes[y1:y2, x1:x2]
        # ## roi 부분의 rgb값 변경하기 (input 이미지 평균 색으로 맞추기)
        # roi[:,:,0] = round(np.array(img_color[:,:,0]).mean())
        # roi[:,:,1] = round(np.array(img_color[:,:,1]).mean())
        # roi[:,:,2] = round(np.array(img_color[:,:,2]).mean())



    # print('shelf coords :\n',coords_bottom)
    # display("boxes2", boxes)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return width, height, coords_bottom, coords_top, boxes

# img_color = cv.imread('C:/flashwoman/EyeCandy/img/source/bookshelf_04.jpg')
# create_bookshelf(img_color)