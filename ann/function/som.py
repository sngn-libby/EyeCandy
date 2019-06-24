from math import *
from .node import *


class som :

    # Recommend to let set raidus=False to auto-caculating the radius
    def __int__(self, height=10, width=10, FV_size=10, PV_size=10, radius=False, learning_rate=0.005):
        self.height = height
        self.width = width
        self.radius = radius if radius else (height+width)/2
        self.total = height * width
        self.learning_rate = learning_rate
        self.nodes = [0] * (self.total)
        self.FV_size = FV_size
        self.PV_size = PV_size

        for i in range(self.height):

            for j in range(self.width):
                self.nodes[(i)*(self.width) + j] = node(FV_size, PV_size, i, j)
                # nodes 의 한 노드에 node가 또 있는건가..?


    # Train_vector format : [ [FV[0], PV[0]], [FV[1], PV[1]], so on .. ]
    def train(self, iterations=1000, train_vector=[[[0.0], [0.0]]]):
        # 변수 선언
        time_constant = iterations / log(self.radius)
        radius_decaying = 0.0
        learning_rate_decaying = 0.0
        influence = 0.0
        stack = [] # Stack for storing best matching unit's index and updated FV and PV
        temp_FV = [0.0] * self.FV_size
        temp_PV = [0.0] * self.PV_size

        for i in range(1, iterations +1):
            print("Iteration number: ", i)
            radius_decaying = self.radius * exp( (-1.0)*i/time_constant )
            learning_rate_decaying = self.learning_rate * exp( (-1.0)*i/time_constant )

            for j in range(len(train_vector)):
                input_FV = train_vector[j][0]
                input_PV = train_vector[j][1]
                best = self.best_match(input_FV)
                stack = []
                for k in range(self.total):
                    dist = self.distance(self.nodes[best], self.nodes[k])

                    if dist < radius_decaying:
                        temp_FV = [0.0] * self.FV_size
                        temp_PV = [0.0] * self.PV_size
                        influence = exp( (-1.0 * (dist**2)) / (2*radius_decaying*i) )

                        for l in range(self.FV_size):
                            # Learning
                            temp_FV[l] = self.nodes[k].FV[l] + \
                                         influence * learning_rate_decaying * (input_FV[l] - self.nodes[k].FV[l])

                        # Push the unit onto stack to update in next interval
                        stack[0:0] = [[[k], temp_FV, temp_PV]]

                for l in range(len(stack)):
                    self.nodes[stack[l][0][0]].FV[:] = stack[l][1][:]
                    self.nodes[stack[l][0][0]].PV[:] = stack[l][2][:]


    # Returns prediction vector
    def predict(self, FV=[0.0]):
        best = self.best_match(FV)
        return self.nodes[best].PV


    # Returns best matching unit's index
    def best_match(self, target_FV=[0.0]):
        minimum = sqrt(self.FV_size) # Minimum distance
        minimum_index = 1 # Minimum distance unit
        temp = 0.0

        for i in range(self.total):
            temp = 0.0
            temp = self.FV_distance(self.nodes[i].FV, target_FV)

            if temp < minimum:
                minimum = temp
                minimum_index = i

        return minimum_index


    def FV_distance(self, FV_1=[0.0], FV_2=[0.0]):
        temp = 0.0

        for j in range(self.FV_size):
            temp = temp + (FV_1[j] - FV_2[j]) ** 2

        temp = sqrt(temp)
        return temp


    def distance(self, node1, node2):
        return sqrt((node1.X - node2.X) ** 2 + (node1.Y - node2.Y) ** 2)


if __name__ == "__main__":
    print( "Initialization...", a = som(5, 5, 2, 1, False, 0.05) )
    print( "Training for the XOR function...", a.train(100, [[[1,0], [1]],\
                                                             [[1,1], [0]],\
                                                             [[0,1], [1]],\
                                                             [[0,0], [0]]]))
    print( "Predictions for the XOR function..." )
    print( "Prediction 0 0,", round(a.predict([0,0])[0]) )
    print( "Prediction 1 0,", round(a.predict([1,0])[0]) )
    print( "Prediction 0 1,", round(a.predict([0,1])[0]) )
    print( "Prediction 1 1,", round(a.predict([1,1])[0]) )







#########################
##      SOM_VER.2      ##
#########################
    # from math import sqrt
# from numpy import (array, unravel_index, nditer, linalg, random, subtract,
#                    power, exp, pi, zeros, arange, outer, meshgrid, dot)
# from collections import defaultdict
# from warnings import warn
#
#
# """
#     Minimalistic implementation of the Self Organizing Maps (SOM).
# """
#
#
# def fast_norm(x):
#     """Returns norm-2 of a 1-D numpy array.
#
#     * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
#     """
#     return sqrt(dot(x, x.T))
#
#
# class MiniSom(object):
#     def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5, decay_function=None, random_seed=None):
#         """
#             Initializes a Self Organizing Maps.
#
#             x,y - dimensions of the SOM
#
#             input_len - number of the elements of the vectors in input
#
#             sigma - spread of the neighborhood function (Gaussian), needs to be adequate to the dimensions of the map.
#             (at the iteration t we have sigma(t) = sigma / (1 + t/T) where T is #num_iteration/2)
#
#             learning_rate - initial learning rate
#             (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T) where T is #num_iteration/2)
#
#             decay_function, function that reduces learning_rate and sigma at each iteration
#                             default function: lambda x,current_iteration,max_iter: x/(1+current_iteration/max_iter)
#
#             random_seed, random seed to use.
#         """
#         if sigma >= x/2.0 or sigma >= y/2.0:
#             warn('Warning: sigma is too high for the dimension of the map.')
#         if random_seed:
#             self.random_generator = random.RandomState(random_seed)
#         else:
#             self.random_generator = random.RandomState(random_seed)
#         if decay_function:
#             self._decay_function = decay_function
#         else:
#             self._decay_function = lambda x, t, max_iter: x/(1+t/max_iter)
#         self.learning_rate = learning_rate
#         self.sigma = sigma
#         self.weights = self.random_generator.rand(x,y,input_len)*2-1 # random initialization
#         for i in range(x):
#             for j in range(y):
#                 self.weights[i,j] = self.weights[i,j] / fast_norm(self.weights[i,j]) # normalization
#         self.activation_map = zeros((x,y))
#         self.neigx = arange(x)
#         self.neigy = arange(y) # used to evaluate the neighborhood function
#         self.neighborhood = self.gaussian
#
#     def _activate(self, x):
#         """ Updates matrix activation_map, in this matrix the element i,j is the response of the neuron i,j to x """
#         s = subtract(x, self.weights) # x - w
#         it = nditer(self.activation_map, flags=['multi_index'])
#         while not it.finished:
#             self.activation_map[it.multi_index] = fast_norm(s[it.multi_index])  # || x - w ||
#             it.iternext()
#
#     def activate(self, x):
#         """ Returns the activation map to x """
#         self._activate(x)
#         return self.activation_map
#
#     def gaussian(self, c, sigma):
#         """ Returns a Gaussian centered in c """
#         d = 2*pi*sigma*sigma
#         ax = exp(-power(self.neigx-c[0], 2)/d)
#         ay = exp(-power(self.neigy-c[1], 2)/d)
#         return outer(ax, ay)  # the external product gives a matrix
#
#     def diff_gaussian(self, c, sigma):
#         """ Mexican hat centered in c (unused) """
#         xx, yy = meshgrid(self.neigx, self.neigy)
#         p = power(xx-c[0], 2) + power(yy-c[1], 2)
#         d = 2*pi*sigma*sigma
#         return exp(-p/d)*(1-2/d*p)
#
#     def winner(self, x):
#         """ Computes the coordinates of the winning neuron for the sample x """
#         self._activate(x)
#         return unravel_index(self.activation_map.argmin(), self.activation_map.shape)
#
#     def update(self, x, win, t):
#         """
#             Updates the weights of the neurons.
#             x - current pattern to learn
#             win - position of the winning neuron for x (array or tuple).
#             t - iteration index
#         """
#         eta = self._decay_function(self.learning_rate, t, self.T)
#         sig = self._decay_function(self.sigma, t, self.T) # sigma and learning rate decrease with the same rule
#         g = self.neighborhood(win, sig)*eta # improves the performances
#         it = nditer(g, flags=['multi_index'])
#         while not it.finished:
#             # eta * neighborhood_function * (x-w)
#             self.weights[it.multi_index] += g[it.multi_index]*(x-self.weights[it.multi_index])
#             # normalization
#             self.weights[it.multi_index] = self.weights[it.multi_index] / fast_norm(self.weights[it.multi_index])
#             it.iternext()
#
#     def quantization(self, data):
#         """ Assigns a code book (weights vector of the winning neuron) to each sample in data. """
#         q = zeros(data.shape)
#         for i, x in enumerate(data):
#             q[i] = self.weights[self.winner(x)]
#         return q
#
#     def random_weights_init(self, data):
#         """ Initializes the weights of the SOM picking random samples from data """
#         it = nditer(self.activation_map, flags=['multi_index'])
#         while not it.finished:
#             self.weights[it.multi_index] = data[self.random_generator.randint(len(data))]
#             self.weights[it.multi_index] = self.weights[it.multi_index]/fast_norm(self.weights[it.multi_index])
#             it.iternext()
#
#     def train_random(self, data, num_iteration):
#         """ Trains the SOM picking samples at random from data """
#         self._init_T(num_iteration)
#         for iteration in range(num_iteration):
#             rand_i = self.random_generator.randint(len(data)) # pick a random sample
#             self.update(data[rand_i], self.winner(data[rand_i]), iteration)
#
#     def train_batch(self, data, num_iteration):
#         """ Trains using all the vectors in data sequentially """
#         self._init_T(len(data)*num_iteration)
#         iteration = 0
#         while iteration < num_iteration:
#             idx = iteration % (len(data)-1)
#             self.update(data[idx], self.winner(data[idx]), iteration)
#             iteration += 1
#
#     def _init_T(self, num_iteration):
#         """ Initializes the parameter T needed to adjust the learning rate """
#         self.T = num_iteration/2  # keeps the learning rate nearly constant for the last half of the iterations
#
#     def distance_map(self):
#         """ Returns the distance map of the weights.
#             Each cell is the normalised sum of the distances between a neuron and its neighbours.
#         """
#         um = zeros((self.weights.shape[0], self.weights.shape[1]))
#         it = nditer(um, flags=['multi_index'])
#         while not it.finished:
#             for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
#                 for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
#                     if ii >= 0 and ii < self.weights.shape[0] and jj >= 0 and jj < self.weights.shape[1]:
#                         um[it.multi_index] += fast_norm(self.weights[ii, jj, :]-self.weights[it.multi_index])
#             it.iternext()
#         um = um/um.max()
#         return um
#
#     def activation_response(self, data):
#         """
#             Returns a matrix where the element i,j is the number of times
#             that the neuron i,j have been winner.
#         """
#         a = zeros((self.weights.shape[0], self.weights.shape[1]))
#         for x in data:
#             a[self.winner(x)] += 1
#         return a
#
#     def quantization_error(self, data):
#         """
#             Returns the quantization error computed as the average distance between
#             each input sample and its best matching unit.
#         """
#         error = 0
#         for x in data:
#             error += fast_norm(x-self.weights[self.winner(x)])
#         return error/len(data)
#
#     def win_map(self, data):
#         """
#             Returns a dictionary wm where wm[(i,j)] is a list with all the patterns
#             that have been mapped in the position i,j.
#         """
#         winmap = defaultdict(list)
#         for x in data:
#             winmap[self.winner(x)].append(x)
#         return winmap
#
# ### unit tests
# '''
# from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
#
#
# class TestMinisom:
#     def setup_method(self, method):
#         self.som = MiniSom(5, 5, 1)
#         for i in range(5):
#             for j in range(5):
#                 assert_almost_equal(1.0, linalg.norm(self.som.weights[i,j]))  # checking weights normalization
#         self.som.weights = zeros((5, 5))  # fake weights
#         self.som.weights[2, 3] = 5.0
#         self.som.weights[1, 1] = 2.0
#
#     def test_decay_function(self):
#         assert self.som._decay_function(1., 2., 3.) == 1./(1.+2./3.)
#
#     def test_fast_norm(self):
#         assert fast_norm(array([1, 3])) == sqrt(1+9)
#
#     def test_gaussian(self):
#         bell = self.som.gaussian((2, 2), 1)
#         assert bell.max() == 1.0
#         assert bell.argmax() == 12  # unravel(12) = (2,2)
#
#     def test_win_map(self):
#         winners = self.som.win_map([5.0, 2.0])
#         assert winners[(2, 3)][0] == 5.0
#         assert winners[(1, 1)][0] == 2.0
#
#     def test_activation_reponse(self):
#         response = self.som.activation_response([5.0, 2.0])
#         assert response[2, 3] == 1
#         assert response[1, 1] == 1
#
#     def test_activate(self):
#         assert self.som.activate(5.0).argmin() == 13.0  # unravel(13) = (2,3)
#
#     def test_quantization_error(self):
#         self.som.quantization_error([5, 2]) == 0.0
#         self.som.quantization_error([4, 1]) == 0.5
#
#     def test_quantization(self):
#         q = self.som.quantization(array([4, 2]))
#         assert q[0] == 5.0
#         assert q[1] == 2.0
#
#     def test_random_seed(self):
#         som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
#         som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
#         assert_array_almost_equal(som1.weights, som2.weights)  # same initialization
#         data = random.rand(100,2)
#         som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
#         som1.train_random(data,10)
#         som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
#         som2.train_random(data,10)
#         assert_array_almost_equal(som1.weights,som2.weights)  # same state after training
#
#     def test_train_batch(self):
#         som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
#         data = array([[4, 2], [3, 1]])
#         q1 = som.quantization_error(data)
#         som.train_batch(data, 10)
#         assert q1 > som.quantization_error(data)
#
#     def test_train_random(self):
#         som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
#         data = array([[4, 2], [3, 1]])
#         q1 = som.quantization_error(data)
#         som.train_random(data, 10)
#         assert q1 > som.quantization_error(data)
#
#     def test_random_weights_init(self):
#         som = MiniSom(2, 2, 2, random_seed=1)
        som.random_weights_init(array([[1.0, .0]]))
        for w in som.weights:
            assert_array_equal(w[0], array([1.0, .0]))



'''