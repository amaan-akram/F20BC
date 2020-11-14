import numpy
from random import random
import pandas



# ##################### Activation Functions ##############################

#return 0 for every value


def null(X):
    return 0

#Return
def sigmoid(X):
    return 1/(1+numpy.exp(-X))


def hyperbolic_Tangent(X):
    return numpy.tanh(X)


def cosine(X):
    return numpy.cos(X)


def gaussian(X):
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




#print(network)

#n_inputs = len(dataset[0]) - 1
#n_outputs = len(set([row[-1] for row in dataset]))
#network = initialize_network(n_inputs, 2, n_outputs)
