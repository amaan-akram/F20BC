import numpy
from random import random


# ############################## Activation Functions ##############################
def null(X):
    return 0


def sigmoid(X):
    return 1 / (1 + numpy.exp(-X))


def hyperbolic_Tangent(X):
    return numpy.tanh(X)


def cosine(X):
    return numpy.cos(X)


def gaussian(X):
    return numpy.exp(-numpy.square(X) / 2)


# Neuron class
# takes in node number - neuron id
# layer type - where the connections exists.
# List of weights
# The neurons activation function
# Each neuron has their bias set to 0
class Neuron:
    def __init__(self, node, layer_type, weights, activationFunc):
        self.node = node
        self.type = layer_type
        self.weights = weights
        self.bias = 0
        self.activationFunc = activationFunc

    # prints the neuron data
    def printNeuron(self):
        print("Neuron : ", self.node, " layer : ", self.type, " weights : ", self.weights, " bias : ", self.bias)

    # computes value of the neuron using the input data
    # returns the value of returned from the activation function
    def compute(self, values):
        totalOfValues = 0
        for i in range(len(self.weights)):
            totalOfValues += self.weights[i] * values[i]
        return self.activationFunc(self.bias + totalOfValues)

    # changes the weights of the neuron
    def changeWeight(self, weights):
        self.weights = weights
        return self.weights


# updates all the weights and biases in the network with new weights
def updateAllWeights(network, weights):
    for layer in network:
        for neuron in layer:
            for w in range(len(neuron.weights)):
                neuron.weights[w] = weights[w]
            neuron.bias = weights[w+1]


# returns the values of the neurons by feeding forward
def forward(network, inp):
    inputs = inp
    for layer in network:
        new_inputs = []
        for neuron in layer:
            neuron = neuron.compute(inputs)
            new_inputs.append(neuron)
        inputs = new_inputs
    return inputs


# Calculates the dimensions of the NN
def dimensions_num(inp, hid, out):
    num_con = 0
    num_bias = 0
    for i in hid:
        num_bias += i
    num_bias = num_bias + out
    connections = [inp]
    for i in hid:
        connections.append(i)
    connections.append(out)
    for i in range(len(connections) - 1):
        num_con += connections[i] * connections[i + 1]
    dimensions = num_con + num_bias
    return dimensions


# Initialises the neural network
# Creates lists of each layer with a list of random weights for each neuron
def createNN(inp, hid, out, activationFunction):
    dim = dimensions_num(inp, hid, out)
    network = []
    hiddenLayers = []
    hidLayer = [Neuron(i, "input - hidden", [random() for _ in range(inp)], activationFunction) for i in range(hid[0])]
    hiddenLayers.append(hidLayer)
    network.append(hidLayer)

    # Creates hidden layer network of layers.
    if len(hid) > 1:
        for i in range(1, len(hid)):
            hidLayer = [Neuron(i, "hidden - hidden", [random() for _ in range(len(hiddenLayers[len(hiddenLayers)-1]))],
                               activationFunction) for i in range(hid[i])]

            hiddenLayers.append(hidLayer)
            network.append(hidLayer)

    # creates the output layer
    outputLayer = [Neuron(i, "hidden - output", [random() for _ in range(len(hiddenLayers[len(hiddenLayers)-1]))],
                          activationFunction) for i in range(out)]

    network.append(outputLayer)
    return network, dim
