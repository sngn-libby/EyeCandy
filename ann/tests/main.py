import numpy as np
import cv2 as cv
from bookshelf.function.contours import contours
# from bookshelf.function.display import display
from bookshelf.function.preprocessing import preprocessing
from ann.function.sompy import SOMFactory, SOM
import pandas as pd


# 1. load image
img = cv.imread('C:/flashwoman/EyeCandy/img/source/img_book_only.png')
img_org = cv.imread('C:/flashwoman/EyeCandy/img/source/bookshelf_04.jpg')
img_res = img_org.copy()
# display('copied',img_res)


# 2. preprocess images
img = preprocessing(img)
img, rect = contours(img,img_org)
# from bookshelf.function.contours_grad import contours_grad
# img, rect = contours_grad(img,img_org)
# print("rect__lt_coord__rb_coord :",rect) # rect에 좌표 저장 [Left_Bottom, Right_Top]
# organizing rect
lb_x=[]; lb_y=[]; rt_x=[]; rt_y=[]; lt_coord=[]; rb_coord=[]; rt_coord=[]; lb_coord=[];

for i in range(len(rect)):

    lb_x.append(rect[i][0][0])
    lb_y.append(rect[i][0][1])
    rt_x.append(rect[i][1][0])
    rt_y.append(rect[i][1][1])
    lt_coord.append([rect[i][0][0],rect[i][1][1]])
    rb_coord.append([rect[i][1][0],rect[i][0][1]])
    rt_coord.append([rect[i][1][0],rect[i][1][1]])
    lb_coord.append([rect[i][0][0],rect[i][0][1]])


# 3. bring mid coordinates
coord = []
for i in range(len(rect)):
    # extracting colour by using mid-coordinates of the book
    x = (list(rect[i][0])[0] + list(rect[i][1])[0])/2
    y = (list(rect[i][0])[1] + list(rect[i][1])[1])/2
    coord.append([x, y])

# print('book_mid_coordinates :\n', coord)


# 4. get 'bgr' from mid-coordinates
bgr_val = []
b_val=[]; g_val=[]; r_val=[]
for i in range(len(coord)):
    ## 1. x, y 좌표값 받아오기
    x, y = coord[i]
    # print(len(coord), i, coord[i], a, img_org.item(round(b), round(a), 0))

    ## 2. float -> int : (.item()은 int값만 받는다)
    x = round(x)
    y = round(y)

    ## 3. 좌표의 BGR 값 받아오기
    bgr_val.append([img_org.item(y, x, 0),
                    img_org.item(y, x, 1),
                    img_org.item(y, x, 2)])

    b_val.append(bgr_val[i][0])
    g_val.append(bgr_val[i][1])
    r_val.append(bgr_val[i][2])

# print(bgr_val[0])
# print('selected_pixel_bgr :\n', bgr_val)


# 5. get book's height
h_val = []
w_val = []
for i in range(len(rect)):
    width = abs(list(rect[i][1])[0] - list(rect[i][0])[0])
    height = abs(list(rect[i][1])[1] - list(rect[i][0])[1])
    h_val.append(height)
    w_val.append(width)


# 6. get position value (left_top)
pos_val = []
for i in range(len(rect)):
    x = list(rect[i][0])[0]
    y = list(rect[i][0])[1]
    pos_val.append([x, y])


# 7. SOM


## 1. preparing data
# print(b_val)
b_val = pd.Series(b_val)
g_val = pd.Series(g_val)
r_val = pd.Series(r_val)
h_val = pd.Series(h_val)
data = pd.concat([b_val,g_val,r_val,h_val], axis=1)
data = data.apply(pd.to_numeric, errors='coerce')
names = ['b','g','r', 'book_height']
# print(data)


## 2. create the SOM network and train it
network = SOMFactory().build(data.values, normalization='var', initialization='pca',
                             component_names=names, mapsize=(6,22)) # , mapsize=(6,-1)
network.train(n_job=1, verbose=False) # , train_rough_len=2, train_finetune_len=5
# print('som mapshape',network.mapshape, network.calculate_map_size(lattice='rect'))

topographic_error = network.calculate_topographic_error()
quantization_error = np.mean(network._bmu[1])
# print (f"Topographic error = {topographic_error}; Quantization error = {quantization_error}")


## 3. Visualization
### 1. component planes view ('b','g','r','book_height')
from ann.function.visualization.mapview import View2D

view2D = View2D(6,6,"rand data",text_size=9)


### 2. U-matrix plot
from ann.function.visualization.umatrix import UMatrixView

umat = UMatrixView(width=6, height=6, title='U-matrix')


## 4. Do K-means clustering on the SOM grid, sweep across k=2 to 15
from ann.function.visualization.hitmap import HitMapView
K = 15 # n_cluster
# K_opt = 12
K_opt = network.cluster(K, 0) # find optimal K
[labels, km, norm_data] = network.cluster(K, K_opt)
hits = HitMapView(9,6, "Clustering", text_size=10)

## 5. Show graphics
# view2D.show(network, col_sz=4, which_dim="all", desnormalize=True)
# umat.show(network)
# hits.show(network)

# print(len(labels))
# print(len(norm_data))
# print(labels)


## 6. Labeling coordinates

bmu_nodes = network.find_k_nodes(data, k=1) # bmu_nodes = 135 X k
# concatenate in one list
bmu_nodes_list = []

for i in bmu_nodes[1]:
    for j in i:
        bmu_nodes_list.append(j)

bmu_nodes_arr = np.asarray(bmu_nodes_list)
# print('bmu_nodes_arr :\n',bmu_nodes_arr)
'''
input : bmu_ind = best matching unit index ---> ouput : corresponding matrix x,y coordinates
calculate coordinates logic : index = x + (y * width)
'''
x_y_ind = network.bmu_ind_to_xy(bmu_nodes_arr)
# print('bmu xy index :\n',x_y_ind, len(x_y_ind))


# 7. current index, target index
# Reference code : sorted(student_tuples, key=lambda student: student[2])
tar_ind = []
cur_ind = []
cur_node_coord = []
for ind, x_y_ind in enumerate(x_y_ind):
    tar_ind.append(x_y_ind[2])
    cur_ind.append(ind)
    temp_x = x_y_ind[1]
    temp_y = x_y_ind[0]
    cur_node_coord.append([temp_x, temp_y])
    # print(ind, x_y_ind[2])

# print('tar_ind : \n',tar_ind, '\n\n')
# print('cur_ind : \n', cur_ind, '\n\n')



# 8. Rearrange Books
from bookshelf.function.create_bookshelf import create_bookshelf

## 1. get bookshelf
mean_book_height = float(np.mean(h_val))
print("mean_booksize : ", mean_book_height)
opt_shelf_height = round(mean_book_height * 1.5)
shelf_width, shelf_height, shelf_coords_bottom, shelf_coords_top, img_boxes = create_bookshelf(img_org, np.array(h_val).max())

# print('shelf_coords_bottom :\n',shelf_coords_bottom)
# print('shelf_coords_bottom access x_y:\n', shelf_coords_bottom[0][0][0], shelf_coords_bottom[0][0][1])
# print('shelf_width :\n',shelf_width)
# print('shelf_height :\n',shelf_height)

## 2. define variables

### define variables
book_slice = []
book_arr = []
roi=[]
num_of_book = round(len(tar_ind)/4)
# print("num_of_book :", num_of_book)
width_per_book = round(shelf_width / num_of_book)
# print("proportioned book width", width_per_book)
boxes = img_boxes.copy()

## 3. paste books on new bookshelf
for n in range(4):

    print(f'######## {n+1} shelf ########')

    ### shelf coord
    # print("shelf_cocords :\n", shelf_coords_bottom)
    n_shelf_x1 = shelf_coords_bottom[n][0][0]
    # print("n_shelf_x1 ", n_shelf_x1)
    n_shelf_y = shelf_coords_bottom[n][0][1]
    tar_x1 = n_shelf_x1
    tar_y1 = n_shelf_y

    length = 0
    # print(np.array(img_org).shape) # (459, 612, 3)

    for i in range(num_of_book):

        if i + (num_of_book*n) < len(rb_coord):

            ### 1. select book to paste
            #### image source coord
            # print('rb coord :\n', rb_coord)
            # print('rb coord access :\n', rb_coord[i])
            [x2, y1] = rb_coord[i + (num_of_book*n)]
            [x1, y1] = lb_coord[i + (num_of_book*n)]
            [x1, y2] = lt_coord[i + (num_of_book*n)]
            [x2, y2] = rt_coord[i + (num_of_book*n)]
            # print('x coords : \n', x1, x2)
            width_real = abs(x2 - x1)
            height_real = int(abs(y2 - y1))

            length += width_real


            if length <= shelf_width:

                if width_real <= width_per_book:
                    print("It's small book")

                else:
                    print("It's quite big book XD")
                    # prop = width_per_book / width_real
                    # print("else clac_prop :", width_real, width_per_book, prop)
                    # height_per_book = int(prop * height_real)
                    # print("proportioned book height", height_per_book)
                    # y2 = y1 + height_per_book

                width_per_book = width_real
                height_per_book = height_real
                # print("proportioned book height", height_per_book)

                ### 2. image slice
                # print('roi y range', y1, y2)
                # print('roi x range', x1, x2)
                roi = img_res[y1:y2, x1:x2]
                print("roi shape :", np.array(roi).shape)

                ### 3. image paste
                if i == 0:

                    print(f"#{i+1} book")

                    # set range
                    # print("tar coord", tar_x1, width_per_book, tar_y1, height_per_book)
                    tar_x2 = tar_x1 + width_per_book
                    tar_y2 = tar_y1 + height_per_book

                    # paste
                    # print("tar_roi y range", tar_y1, tar_y2)
                    # print("tar_roi x range", tar_x1, tar_x2)
                    # print(tar_y2 - tar_y1)
                    # print(np.array(img_boxes).shape)
                    print("target roi shape :", np.array(img_boxes[tar_y1:tar_y2, tar_x1:tar_x2]).shape)
                    img_boxes[tar_y1:tar_y2, tar_x1:tar_x2] = roi

                else:

                    print(f"#{i+1} book")

                    # set range
                    tar_x1 += width_per_book
                    tar_x2 = tar_x1 + width_per_book
                    tar_y2 = tar_y1 + height_per_book
                    # print("tar_roi y range", tar_y1, tar_y2)
                    # print("tar_roi x range", tar_x1, tar_x2)
                    print("target roi shape :", np.array(img_boxes[tar_y1:tar_y2, tar_x1:tar_x2]).shape)

                    ### paste
                    img_boxes[tar_y1:tar_y2, tar_x1:tar_x2] = roi # * prop

            # elif ( n==3 ) and ( i != num_of_book ):
            #     print("you need to buy extra shelf... you have too many book to arrange properly :(")

        else:

            print("Oh! this shelf looks full...!\nLet's stack on next one :)")

# 9. image post-processing
img_boxes = cv.flip(img_boxes, 0)


# 10. show and write image!!!!
from bookshelf.function.img_write import img_write

# cv.imshow("origin image", img_org)
# cv.imshow("roi", roi)
# cv.imshow("img_boxes", img_boxes)
# img_write('final_result', img_boxes)
# cv.waitKey(0)

img_write("output", img_boxes)
print("codes executed properly")
# 10. ~~Session End~~
# cv.destroyAllWindows()




# KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
#        n_clusters=11, n_init=10, n_jobs=None, precompute_distances='auto',
#        random_state=None, tol=0.0001, verbose=0)

