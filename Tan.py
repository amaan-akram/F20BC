import ANN as ann
import PSO as pso
import data_prep as dp


# create the network, number of input neurons, an array of hidden layer neurons and a number of output neurons.
# returns the network structure with the number of dimensions the network has
network, dim = ann.createNN(1, [7], 1, ann.hyperbolic_Tangent)
# separates the input data from the output data given a file.
inp, output = dp.prepare_data("Data/1in_tanh.txt")
# run PSO to find the best position
BEST_OVERALL = pso.PSO(iterations=200, swarm_size=100, bounds=[-20, 20], dim=dim, inf_num=50, max_vel=0.85,
                       max_pb=3, max_ib=0.1, max_gb=1, inp=inp, output=output, network=network)

# set up the predicted values list.
predicted_values = []
# update the weights to the best position values generated from the pso
ann.updateAllWeights(network, BEST_OVERALL)
# loop through the inputs of the file and insert them into the ann then plot the graph
for i in inp:
    predicted_values.append(ann.forward(network, i))
pso.plotGraph(inp, output, predicted_values, "Tanh")
