import numpy
from random import random
import pandas


# ##################### Activation Functions ##############################

#return 0 for every value


def null(X):
    return 0

#Return
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

def new_network(inp, h, o, activationfunc, data):


    network = list()
    hidden_layer = [{'weights':[random() for i in range(inp + 1)]} for i in range(h)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(h + 1)]} for i in range(o)]
    network.append(output_layer)
    return forward_propagate(network, data, activationfunc)


def newnew_network(inp, h, o):

    print (layers)
    weights = list()

    for i in range(len(layers)-1):
        weight = numpy.random.rand(layers[i], layers[i+1])
        weights.append(weight)

    return weights



# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation


def forward_propagate(network, row, activationfunc):
    activations = row

    for w in network:
        for i in w:
            for j in i:
                # preform matrix mult between activations and weights
                net_inputs = numpy.dot(activations, j)

        #cal acvtivations
        activations = activationfunc(net_inputs)

    return activations



data = [1, 1,2,1]
#network = new_network(1,5,2, Gaussian, data)
new_network1 = newnew_network(2,[1,1,1,1,1,1],1)



print(new_network1)
print(forward_propagate(new_network1, data, sigmoid))

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
        print("Neuron : ", self.node, " layer : ", self.type, " weights : ", self.weights, " bias : ", self.bias)

    def compute(self, values):
        totalOfValues = 0
        for i in range(len(self.weights)):
            totalOfValues += self.weights[i] * values[i]
        return self.activationFunc(self.bias + totalOfValues)

    def changeWeight(self, weights):
        self.weights = weights
        return self.weights


def updateAllWeights(network, weights):
    for layer in network:
        for neuron in layer:
            for w in range(len(neuron.weights)):
                neuron.weights[w] = weights[w]
            neuron.bias = weights[w+1]


def forward(network, inp):
    inputs = inp
    for layer in network:
        new_inputs = []
        for neuron in layer:
            neuron = neuron.compute(inputs)
            new_inputs.append(neuron)
        inputs = new_inputs
    return inputs


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


def createNN(inp, hid, out, activationFunction):
    network = []
    hiddenLayers = []
    hidLayer = [Neuron(i, "input - hidden", [random() for _ in range(inp)], activationFunction) for i in range(hid[0])]
    hiddenLayers.append(hidLayer)
    network.append(hidLayer)

    if len(hid) > 1:
        for i in range(1, len(hid)):
            hidLayer = [Neuron(i, "hidden - hidden", [random() for _ in range(len(hiddenLayers[len(hiddenLayers)-1]))],
                               activationFunction) for i in range(hid[i])]

            hiddenLayers.append(hidLayer)
            network.append(hidLayer)

    outputLayer = [Neuron(i, "hidden - output", [random() for _ in range(len(hiddenLayers[len(hiddenLayers)-1]))],
                          activationFunction) for i in range(out)]

    network.append(outputLayer)
    print(network)
    return network


#network = createNN(5, [3,2,3], 4, sigmoid)
#updateAllWeights(network, weights=0)
#print(forward(network, [1, 2, 3, 4, 5]))
#print(forward(network, [1, 2, 4]))

"""net = createNN(1, [3, 3, 3], 1, sigmoid)
print("Dimensions : ", dimensions_num(1, [3, 3, 3], 1))
print(net[3][0].printNeuron())
print(net[3][0].weights[1])
print(net[0][0].bias)"""

#n_inputs = len(dataset[0]) - 1
#n_outputs = len(set([row[-1] for row in dataset]))
#network = initialize_network(n_inputs, 2, n_outputs)
