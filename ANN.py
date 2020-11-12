import numpy
from random import seed
from random import random
import matplotlib.pyplot as plt


# ##################### Activation Functions ##############################
def Null(X):
    return 0


def sigmoid(X):
    return 1/(1+numpy.exp(-X))


def Hyperbolic_Tangent(X):
    return numpy.tanh(X)


def Cosine(X):
    return numpy.cos(X)


def Gaussian(X):
    return numpy.exp(-numpy.square(X)/2)


# i = input layer, h = hidden layers, o = output layers
def new_network(inp, h, o):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(inp + 1)]} for i in range(h)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(h + 1)]} for i in range(o)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation


def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
           [{'weights': [0.2550690257394217, 0.49543508709194095]},
            {'weights': [0.4494910647887381, 0.651592972722763]}]]
row = [1, 0, None]
output = forward_propagate(network, row)
print(output)
