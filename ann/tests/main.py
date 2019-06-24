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
for i in range(len(rect)):
    height = abs(list(rect[i][0])[1] - list(rect[i][1])[1])
    h_val.append(height)

# 6. get position value
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
                             component_names=names)
network.train(n_job=1, verbose=False, train_rough_len=2, train_finetune_len=5)

topographic_error = network.calculate_topographic_error()
quantization_error = np.mean(network._bmu[1])
print (f"Topographic error = {topographic_error}; Quantization error = {quantization_error}")

# 3. component planes view
from ann.function.visualization.mapview import View2D

view2D = View2D(6,6,"rand data",text_size=12)
view2D.show(network, col_sz=4, which_dim="all", desnormalize=True)

# 4. U-matrix plot
from ann.function.visualization.umatrix import UMatrixView

umat = UMatrixView(width=5, height=5, title='U-matrix')
umat.show(network)


#
# # 5. Do K-means clustering on the SOM grid, sweep across k=2 to 20
from ann.function.visualization.hitmap import HitMapView
K = 20 # stop at this k for SSE sweep
K_opt = 18 # optimal K ?
[labels, km, norm_data] = network.cluster(K, K_opt)
hits = HitMapView(6,6, "Clustering", text_size=12)
a=hits.show(network)
a
print(km)
