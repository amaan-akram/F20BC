import numpy
from random import random
import pandas


# ##################### Activation Functions ##############################
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


# i = input layer,
# h = hidden layers,
# o = output layers,
# activationfunc = (pick one of the above),
# data= array of inputs
"""def new_network(inp, h, o, activationfunc, data):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(inp + 1)]} for i in range(h)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(h + 1)]} for i in range(o)]
    network.append(output_layer)
    return forward_propagate(network, data, activationfunc)

# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation


def forward_propagate(network, row, activationfunc):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = activationfunc(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs




data = [1, 1,2,1, None]
network = new_network(1,5,2, gaussian, data)
print(network)
"""

class Neuron:
    def __init__(self, node, layer_type, weights, activationFunc):
        self.node = node
        self.type = layer_type
        self.weights = weights
        self.bias = 0
        self.activationFunc = activationFunc

    def printNeuron(self):
        print("Neuron : ", self.node, " layer : ", self.type, " weights : ", self.weights)

    def compute(self, values):
        totalOfValues = 0
        for i in range(len(self.weights)):
            totalOfValues += self.weights[i] * values[i]
        return self.activationFunc(self.bias + totalOfValues)


def forward(network, inp):
    inputs = inp
    for layer in network:
        new_inputs = []
        for neuron in layer:
            neuron = neuron.compute(inputs)
            new_inputs.append(neuron)
        inputs = new_inputs
    return inputs


def createNN(inp, hid, out, activationFunction):
    network = []
    hiddenLayers = []
    hidLayer = [Neuron(i, "input - hidden", [random() for _ in range(inp)], activationFunction) for i in range(hid[0])]
    hiddenLayers.append(hidLayer)
    network.append(hidLayer)
    for i in hidLayer:
        i.printNeuron()

    if len(hid) > 1:
        for i in range(1, len(hid)):
            hidLayer = [Neuron(i, "hidden - hidden", [random() for _ in range(len(hiddenLayers[len(hiddenLayers)-1]))],
                               activationFunction) for i in range(hid[i])]

            for j in hidLayer:
                j.printNeuron()
            hiddenLayers.append(hidLayer)
            network.append(hidLayer)

    outputLayer = [Neuron(i, "hidden - output", [random() for _ in range(len(hiddenLayers[len(hiddenLayers)-1]))],
                          activationFunction) for i in range(out)]

    for i in outputLayer:
        i.printNeuron()

    network.append(outputLayer)
    return network


network = createNN(5, [3,2,3], 4, sigmoid)
print(forward(network, [1, 2, 3, 4, 5]))
print(forward(network, [1, 2, 4]))

#print(network)

#n_inputs = len(dataset[0]) - 1
#n_outputs = len(set([row[-1] for row in dataset]))
#network = initialize_network(n_inputs, 2, n_outputs)
