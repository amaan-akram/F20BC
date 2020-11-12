import numpy
from random import seed
from random import random
import matplotlib.pyplot as plt

# i = input layer, h = hidden layers, o = output layers
def new_network(i, h, o):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(i + 1)]} for i in range(h)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(h + 1)]} for i in range(o)]
    network.append(output_layer)
    return network


seed(1)
network = new_network(2, 1, 2)
for layer in network:
    print(layer)







###################### Activation Functions ##############################
def Null (X):
    return 0

def sigmoid(X):
    return 1/(1+numpy.exp(-X))

def Hyperbolic_Tangent(X):
    return numpy.tanh(X)

def Cosine (X):
    return numpy.cos(X)

def Gaussian(X):
    return numpy.exp(-numpy.square(X)/2)