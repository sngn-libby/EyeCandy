# Reference Page of SOM clustering examples
#  : https://github.com/annoviko/pyclustering/blob/master/pyclustering/nnet/examples/som_examples.py
#  : https://github.com/annoviko/pyclustering/blob/master/pyclustering/nnet/som.py
#  : https://github.com/jonathandunn/c2xg/blob/3fcf36fd2d97985b54e09d4dbec36ce018f5a45a/c2xg/modules/clustering/!README.md
# Reference Image
#  : https://github.com/annoviko/pyclustering/blob/master/docs/img/target_som_processing.png
# Reference to Visualize
#  : https://github.com/Dustin21/Unsupervised_Ensemble_Learning

import random
import cv2 as cv
from bookshelf.function.contours import contours
from bookshelf.function.preprocessing import preprocessing
from bookshelf.function.display import display
from pyclustering.nnet.som import som, type_conn, type_init, som_parameters
from ann.function.sompy import SOMFactory
import numpy as np
import pandas as pd
import glob
import os



# 1. load image
img = cv.imread('C:/flashwoman/Object-detection/EyeCandy/img/img_book_only.png')
img_org = cv.imread('C:/flashwoman/Object-detection/EyeCandy/img/bookshelf_04.jpg')

# 2. preprocess images
img = preprocessing(img)
img, rect = contours(img,img_org)

# 3. bring mid coordinates
coord = []
for i in range(len(rect)):
    # extracting colour by using mid-coordinates of the book
    x = (list(rect[i][0])[0] + list(rect[i][1])[0])/2
    y = (list(rect[i][0])[1] + list(rect[i][1])[1])/2
    coord.append([x, y])

print('book_mid_coordinates :\n', coord)

# 4. get 'bgr' from coordinates
bgr_val = []
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

print('selected_pixel_bgr :\n', bgr_val)


## 5. SOM
sample = bgr_val

# create SOM parameters
parameters = som_parameters() # initialization of initial neuron weights, radius, rateOfLearning, autostop

# create self-organized feature map with size 7x7
rows = 6  # five rows
cols = 6  # five columns
structure = type_conn.honeycomb  # each neuron has max. four neighbors.
network = som(rows, cols, structure, parameters)


# train network on sample during 100 epochs.
network.train(sample, 50)

# simulate trained network using randomly modified point from input dataset.
index_point = random.randint(0, len(sample) - 1)
point = sample[index_point]  # obtain randomly point from data
point[0] += random.random() * 0.2  # change randomly X-coordinate
point[1] += random.random() * 0.2  # change randomly Y-coordinate
index_winner = network.simulate(point)

# check what are objects from input data are much close to randomly modified.
index_similar_objects = network.capture_objects[index_winner]

# neuron contains information of encoded objects
print("Point '%s' is similar to objects with indexes '%s'." % (str(point), str(index_similar_objects)))
print("Coordinates of similar objects:")
for index in index_similar_objects: print("\tPoint:", sample[index])

print(network.capture_objects)

# result visualization:
# show distance matrix (U-matrix).
network.show_distance_matrix()
# show density matrix (P-matrix).
network.show_density_matrix(surface_divider=5.0) # default 20.0
# show winner matrix.
network.show_winner_matrix()
# show self-organized map.
network.show_network(awards=True, belongs=True, coupling=True, dataset=True)



cv.destroyAllWindows()


