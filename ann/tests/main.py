import numpy as np
import cv2 as cv
from bookshelf.function.contours import contours
from bookshelf.function.preprocessing import preprocessing
from ann.function.sompy import SOMFactory, SOM
import pandas as pd
import glob
import os


# 1. load image
img = cv.imread('C:/flashwoman/Object-detection/EyeCandy/img/img_book_only.png')
img_org = cv.imread('C:/flashwoman/Object-detection/EyeCandy/img/bookshelf_04.jpg')

# 2. preprocess images
img = preprocessing(img)
img, rect = contours(img,img_org)       # rect에 좌표 저장 [Left_Top, Right_Bottom]
# organizing rect
lt_x=[]; lt_y=[]; rb_x=[]; rb_y=[]; lt_coord=[]; rb_coord=[];lb_coord=[];
for i in range(len(rect)):
    lt_x.append(list(rect[i][0])[0])
    lt_y.append(list(rect[i][0])[1])
    rb_x.append(list(rect[i][1])[0])
    rb_y.append(list(rect[i][1])[1])
    lt_coord.append([lt_x,lt_y])
    rb_coord.append([rb_x,rb_y])
    lb_coord.append([lt_x,rb_y])



# 3. bring mid coordinates
coord = []
for i in range(len(rect)):
    # extracting colour by using mid-coordinates of the book
    x = (list(rect[i][0])[0] + list(rect[i][1])[0])/2
    y = (list(rect[i][0])[1] + list(rect[i][1])[1])/2
    coord.append([x, y])

print('book_mid_coordinates :\n', coord)

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
print(bgr_val[0])

print('selected_pixel_bgr :\n', bgr_val)

# 5. get book's height
h_val = []
w_val = []
for i in range(len(rect)):
    width = abs(list(rect[i][0])[0] - list(rect[i][1])[0])
    height = abs(list(rect[i][0])[1] - list(rect[i][1])[1])
    h_val.append(height)
    w_val.append(width)

# 6. get position value (left_top)
pos_val = []
for i in range(len(rect)):
    x = list(rect[i][0])[0]
    y = list(rect[i][0])[1]
    pos_val.append([x, y])


## 5. SOM

# 1. preparing data
print(b_val)
b_val = pd.Series(b_val)
g_val = pd.Series(g_val)
r_val = pd.Series(r_val)
h_val = pd.Series(h_val)
data = pd.concat([b_val,g_val,r_val,h_val], axis=1)
data = data.apply(pd.to_numeric, errors='coerce')
names = ['b','g','r', 'book_height']
print(data)

# 2. create the SOM network and train it
network = SOMFactory().build(data.values, normalization='var', initialization='pca',
                             component_names=names, mapsize=(6,22)) # , mapsize=(6,-1)
network.train(n_job=1, verbose=False) # , train_rough_len=2, train_finetune_len=5
print('som mapshape',network.mapshape, network.calculate_map_size(lattice='rect'))

topographic_error = network.calculate_topographic_error()
quantization_error = np.mean(network._bmu[1])
print (f"Topographic error = {topographic_error}; Quantization error = {quantization_error}")


# 3. component planes view ('b','g','r','book_height')
from ann.function.visualization.mapview import View2D

view2D = View2D(6,6,"rand data",text_size=9)


# 4. U-matrix plot
from ann.function.visualization.umatrix import UMatrixView

umat = UMatrixView(width=6, height=6, title='U-matrix')


# 5. Do K-means clustering on the SOM grid, sweep across k=2 to 15
from ann.function.visualization.hitmap import HitMapView
K = 15 # n_cluster
K_opt = 12
# K_opt = network.cluster(K, 0) # find optimal K
[labels, km, norm_data] = network.cluster(K, K_opt)
hits = HitMapView(9,6, "Clustering", text_size=10)

# 6. Show graphics
# view2D.show(network, col_sz=4, which_dim="all", desnormalize=True)
# umat.show(network)
# hits.show(network)

print(len(labels))
print(len(norm_data))
print(labels)

# 6. Labeling coordinates

bmu_nodes = network.find_k_nodes(data, k=1) # bmu_nodes = 135 X k
# concatenate in one list
bmu_nodes_list = []

for i in bmu_nodes[1]:
    for j in i:
        bmu_nodes_list.append(j)

bmu_nodes_arr = np.asarray(bmu_nodes_list)
print('bmu_nodes_arr :\n',bmu_nodes_arr)
'''
input : bmu_ind = best matching unit index ---> ouput : corresponding matrix x,y coordinates
calculate coordinates logic : index = x + (y * width)
'''
x_y_ind = network.bmu_ind_to_xy(bmu_nodes_arr)
print('bmu xy index :\n',x_y_ind, len(x_y_ind))


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
    print(ind, x_y_ind[2])

print('tar_ind : \n',tar_ind, '\n\n')
print('cur_ind : \n', cur_ind, '\n\n')



## 6. Rearrange Books
from bookshelf.function.create_bookshelf import create_bookshelf

# 1.
width, height, coords = create_bookshelf(img_org)

print(coords)



print('cur_node_coord : \n',cur_node_coord)
print('lb_coord : \n',lb_coord)

cv.destroyAllWindows()

# KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
#        n_clusters=11, n_init=10, n_jobs=None, precompute_distances='auto',
#        random_state=None, tol=0.0001, verbose=0)

