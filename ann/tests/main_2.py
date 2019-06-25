import cv2 as cv
from bookshelf.function.contours import contours
from bookshelf.function.preprocessing import preprocessing


# 1. load image
img = cv.imread('C:/flashwoman/Object-detection/EyeCandy/img/img_book_only.png')
img_org = cv.imread('C:/flashwoman/Object-detection/EyeCandy/img/bookshelf_04.jpg')

# 2. preprocess images
img = preprocessing(img)
img, rect = contours(img,img_org)       # rect에 좌표 저장 [Left_Top, Right_Bottom]
# organizing rect
lt_x=[]; lt_y=[]; rb_x=[]; rb_y=[]; lt_coord=[]; rb_coord=[]
for i in range(len(rect)):
    lt_x.append(list(rect[i][0])[0])
    lt_y.append(list(rect[i][0])[1])
    rb_x.append(list(rect[i][1])[0])
    rb_y.append(list(rect[i][1])[1])
    lt_coord.append([lt_x,lt_y])
    rb_coord.append([rb_x,rb_y])





## 6. Rearrange Books
from bookshelf.function.create_bookshelf import create_bookshelf

# 1.
width, coords = create_bookshelf(img_org)
print(coords)
print(width)

# Rearrange books
book_slice = []
book_arr = []
j = 0
for i in range(len(rb_coord)):

    num_of_book = len(tar_ind[i])/4
    width_per_book = width / (len(tar_ind[i])/4)
    height_per_book = width_per_book * h_val[i] / w_val[i]
    # cur_node_coord
    [x2, y1] = rb_coord[i]
    [x1, y1] = lb_coord[i]
    width_real = x2 - x1
    prop = width_real / width_per_book
    y2 = y1 + height_per_book
    src = img_org.copy()
    book_slice.append(src[y1:y2, x1:x2])

    if i < num_of_book :


    elif i = num_of_book :
        j += 1



