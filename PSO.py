import random
import ANN as ann
import data_prep as dp
import matplotlib.pyplot as plot
import math


# Class for the particle
# When initialised it will take a position that will be of the dimensions the ann provides excluding the input layer.
# Num is to keep track of the particle number - unique identifier.

# Velocity is init randomly and the error value (fitness value) of each particle is init to infinite as we want to find
# the lowest possible value
# Each particle includes a list of informants
class Particle:
    def __init__(self, position, num):
        self.position = position
        self.velocity = [random.uniform(0, 1) for _ in range(len(self.position))]
        self.error = float("inf")
        self.best_position = self.position
        self.best_error = self.error
        self.informants = []
        self.num = num

    # in order to to find the best informant position to update the velocity we need to loop through the informants list
    # and find the best error value for out of all the informants. Once found the position is then returned.
    def bestInformant(self):
        best = float('inf')
        best_inf_pos = self.position
        for inf in self.informants:
            if best > inf.best_error:
                best_inf_pos = inf.best_position
        return best_inf_pos

    # In order for PSO to learn the particles need to move for this we need to update the velocity -
    # This takes in most of the hyperparameters that are necessary for PSO
    def calc_velocity(self, global_best_position, informant_position, max_vel, max_pb, max_ib, max_gb):
        # This is velocity equation for calculating a new velocity
        # v(t + 1) = (w * v(t)) + (c1 * r1 * (p(t) – x(t)) + (c2 * r2 * (g(t) – x(t))

        # Loop through all the positions
        for i in range(len(self.position)):
            # gets three random numbers for the equation
            # The constants are the proportions set out before the algorithm is executed
            r1 = random.random()
            r2 = random.random()
            r3 = random.random()
            w = max_vel
            c1 = max_pb
            c2 = max_ib
            c3 = max_gb
            # calculate velocity
            new_vel = ((w * self.velocity[i]) +
                       ((c1*r1) * (self.best_position[i] - self.position[i])) +
                       ((c2*r2) * (informant_position[i] - self.position[i])) +
                       ((c3*r3) * (global_best_position[i] - self.position[i])))

            # change the velocity
            self.velocity[i] = new_vel

    # Moves the particle after the new velocity is calculated.
    def update_position(self):
        for i in range(len(self.position)):
            self.position[i] = self.position[i] + self.velocity[i]

    # Fitness function that makes use of the mean squared error. This is the loss function for PSO
    # Calculates the current error for a particle after the weights and biases have been updated with the positions of
    # the particle
    def fitness(self, inp, outp):
        x = self.position
        ann.updateAllWeights(network, x)
        values = []
        for i in inp:
            result = ann.forward(network, [i])
            values.append(result)
        total = 0

        # mean squared error
        for predValue, realValue in zip(values, outp):
            total += (predValue[0] - realValue) ** 2
        self.error = total / len(values)


# Loss function extra chunk
# for i in range(len(values)):
# total += np.mean((values[i][0] - outp[i]) ** 2)
# self.error = total / len(values)
# return values

# Main PSO method
# Takes in iterations, number of particles, bounds for the random locations of the particles and
# the portions of velocity, best particle, informant, global positions
def PSO(iterations, swarm_size, dim, bounds, max_vel, max_pb, max_ib, max_gb):
    # Init GLOBAL best and err as empty and infinite
    GLOBAL_BEST_POSITION = []
    GLOBAL_BEST_ERROR = float("inf")
    i = 0
    # Create a swarm of particles with random positions.
    # The number of particles are set out before algorithm starts - swarmsize
    swarm = [Particle([random.uniform(bounds[0], bounds[1]) for k in range(dim)], num=j) for j in range(swarm_size)]
    # Loop through all the iterations
    while i < iterations:
        # empty the informants list on the next iteration
        for particle in swarm:
            particle.informants = []

        # Loop through each particle and append a random particle to another particles informant list
        # Each particle will have a random number of particles between 1 and the swarm size
        for particle in swarm:
            rand = []
            for j in range(0, random.randint(1, swarm_size)):
                n = random.randint(0, swarm_size-1)
                rand.append(n)
            for j in rand:
                particle.informants.append(swarm[j])

        # Main section of the algorithm - Loops through all the particles and gets the fitness for each one.
        # It will then compare it to the particles personal best fitness followed by the global best
        # If the there is a new best the new position and err will be used.
        for particle in swarm:
            particle.fitness(inp, output)

            if particle.best_error > particle.error:
                particle.best_error = particle.error
                particle.best_position = particle.position

            if particle.error < GLOBAL_BEST_ERROR:
                GLOBAL_BEST_POSITION.clear()
                GLOBAL_BEST_ERROR = particle.error
                for pos in range(len(particle.position)):
                    GLOBAL_BEST_POSITION.append(particle.position[pos])

        print("ITER : ", i, " Global best error = ", GLOBAL_BEST_ERROR)

        # Change the velocity and update the position of each particle.
        for particle in swarm:
            particle.calc_velocity(GLOBAL_BEST_POSITION, particle.bestInformant(), max_vel, max_pb, max_ib, max_gb)
            particle.update_position()

        # increment
        i += 1

    return GLOBAL_BEST_POSITION
    # out1 = original   out2 = result


# used to plot and compare the file data and the input data on the ann with pso
def plotGraph(inp1, out1, out2):
    plot.plot(inp1, out1)
    plot.plot(inp1, out2)
    plot.show()


# create the network, number of input neurons, an array of hidden layer neurons and a number of output neurons.
# returns the network structure with the number of dimensions the network has
network, dim = ann.createNN(1, [7], 1, ann.hyperbolic_Tangent)
# separates the input data from the output data given a file.
inp, output = dp.prepare_data("Data/1in_linear.txt")

BEST_OVERALL = PSO(iterations=200, swarm_size=100, bounds=[-20, 20], dim=dim, max_vel=0.85, max_pb=math.pi, max_ib=0.1, max_gb=(4-math.pi))

# set up the predicted values list.
predicted_values = []
# update the weights to the best position values generated from the pso
ann.updateAllWeights(network, BEST_OVERALL)

# loop through the inputs of the file and insert them into the ann then plot the graph
for i in inp:
    predicted_values.append(ann.forward(network, [i]))
plotGraph(inp, output, predicted_values)
