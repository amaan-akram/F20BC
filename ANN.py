import numpy
from random import random
import pandas



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


# i = input layer,
# h = hidden layers,
# o = output layers,
# activationfunc = (pick one of the above),
# data= array of inputs
def new_network(inp, h, o, activationfunc, data):
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
network = new_network(1,5,2, Gaussian, data)
print(network)



#print(network)

#n_inputs = len(dataset[0]) - 1
#n_outputs = len(set([row[-1] for row in dataset]))
#network = initialize_network(n_inputs, 2, n_outputs)