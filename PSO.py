import random
import math
from random import seed
import numpy as np
import ANN as ann
import data_prep as dp
import matplotlib.pyplot as plot

particle_size = 100  # number of particles
iterations = 200  # max number of iterations
w = 0.85  # inertia constant
c1 = 1  # cognative constant
c2 = 2  # social constant


class Particle:
    def __init__(self, position):
        self.position = position
        self.velocity = [random.uniform(0, 1) for _ in range(len(self.position))]
        self.error = float("inf")
        self.best_position = self.position
        self.best_error = self.error
        self.informant = []

    def bestInformant(self):
        best = float('inf')
        for value in self.informant:
            print(value.best_error)
            if best > value.best_error:
                best = value.best_position
        return best

    def calc_velocity(self, global_best_particle_position):
        for i in range(len(self.position)):
            r1 = random.random()
            r2 = random.random()
            new_vel = ((w * self.velocity[i]) + c1 * r1 * (self.best_position[i] - self.position[i]) + (
                        c2 * r2 * global_best_particle_position[i] - self.position[i]))
            self.velocity[i] = new_vel

    def update_position(self):
        for i in range(len(self.position)):
            self.position[i] = self.position[i] + self.velocity[i]

    def fitness(self, inp, outp):
        x = self.position
        # update weights
        ann.updateAllWeights(network, x)
        # this is the loss function
        values = []
        for i in inp:
            result = ann.forward(network, [i])
            values.append(result)

        # ANN results
        total = 0

        for predValue, realValue in zip(values, outp):
            total += (predValue[0] - realValue) ** 2
        self.error = total / len(values)

        # for i in range(len(values)):
    #  total += np.mean((values[i][0] - outp[i]) ** 2)
    # self.error = total / len(values)
    # return values


def PSO(iterations, swarm_size, bounds):
    GLOBAL_BEST_POSITION = []
    GLOBAL_BEST_ERROR = float("inf")
    i = 0
    swarm = [Particle([random.uniform(bounds[0], bounds[1]) for i in range(dim)]) for _ in range(swarm_size)]
    while i < iterations:
        # get informants
        rand = []
        for j in range(0, (len(swarm) // 2)):
            n = random.randint(0, len(swarm) - 1)
            rand.append(n)

        for particle in swarm:
            for j in rand:
                particle.informant.append(swarm[j])

        for particle in swarm:
            particle.update_position()
            particle.fitness(inp, outp)

            if particle.best_error > particle.error:
                particle.best_error = particle.error
                particle.best_position = particle.position

            if particle.error < GLOBAL_BEST_ERROR:
                GLOBAL_BEST_ERROR = particle.error
                GLOBAL_BEST_POSITION = particle.position

            particle.calc_velocity(particle.bestInformant())

        i += 1

        # out1 = original   out2 = result


def plotGraph(inp1, out1, out2):
    plot.plot(inp1, out1)
    plot.plot(inp1, out2)
    plot.show()


network = ann.createNN(1, [3, 3, 3], 1, ann.hyperbolic_Tangent)
dim = ann.dimensions_num(1, [3, 3, 3], 1)  # MAKE SURE TO CHANGE THIS
inp, outp = dp.prepare_data("Data/1in_cubic.txt")

PSO(iterations=10, swarm_size=10, bounds=[-10, 10])
predicted_values = []
for i in inp:
    predicted_values.append(ann.forward(network, [i]))
plotGraph(inp, outp, predicted_values)

'''
class Particle:
    def __init__(self, position, velocity, num):
        self.position = position
        self.pBest = self.position
        self.pBest_value = float('inf')
        self.velocity = velocity
        self.id = num
        self.informants = []
        self.informantsBest = self.position

    def particlePos(self):
        print("Particle ", self.id, " at position : ", self.position, " and PB is ", self.pBest,
              " with velocity ", self.velocity)

    def bestInformant(self):
        best = float('inf')
        for i in self.informants:
            if best > i.pBest_value:
                best = i.pBest_value
                self.informantsBest = i.pBest

    def newVelocity(self, vel_const, g_best, i_best, p_best, i):
        new_velocity = 0
        self.bestInformant()
        for j in range(i):
            r1 = random.random()
            r2 = random.random()
            new_velocity = vel_const * self.velocity + (p_best * r1 * (self.pBest[j] - self.position[j])) + \
                           (i_best * r2 * (self.informantsBest[j] - self.position[j]))

        return new_velocity


def fitness(particle, inp, exp):
    x = particle.position
    # update weights
    ann.updateAllWeights(network, x)
    # this is the loss function
    values = []
    for i in inp:
        result = ann.forward(network, [i])
        values.append(result)

    # ANN results
    total = 0
    for i in range(len(values)):
        total += np.mean((values[i][0] - exp[i]) ** 2)
    return total / len(values), values


def PSO(swarm_size, vel_const, p_best, i_best, g_best, max_iter, dimensions, bounds, file):
    global_values = []
    inputs, exp = dp.prepare_data(file)
    g_best_value = float('inf')
    g_best_pos = np.array([float('inf'), float('inf')])
    arr = [Particle([random.uniform(bounds[0], bounds[1]) for i in range(dimensions)], random.uniform(0, 1), i) for i in
           range(swarm_size)]
    iter = 0

    while iter < max_iter:
        # get informants
        rand = []
        for i in range(0, (len(arr) // 2)):
            n = random.randint(0, len(arr) - 1)
            rand.append(n)

        for particle in arr:
            for i in rand:
                particle.informants.append(arr[i])

        # update velocity
        for particle in arr:
            particle_fitness, values = fitness(particle, inputs, exp)
            global_values = values
            if particle.pBest_value > particle_fitness:
                particle.pBest_value = particle_fitness
                particle.pBest = particle.position

            if g_best_value > particle_fitness:
                g_best_value = particle_fitness
                g_best_pos = particle.position

            for i in range(len(particle.position)):
                newVel = particle.newVelocity(vel_const=vel_const, p_best=p_best, i_best=i_best, g_best=g_best,
                                              i=dimensions)
                particle.position[i] = particle.position[i] + newVel

        iter += 1
        print("Print", global_values)
    plots.plotPSO(inputs, global_values, inputs, exp)


network = ann.createNN(1, [2,3,4], 1, ann.hyperbolic_Tangent)

f = "Data/1in_cubic.txt"

PSO(swarm_size=30, vel_const=0.8, p_best=0.5, i_best=0.2, g_best=0.3, max_iter=20,
    dimensions=ann.dimensions_num(1, [3, 3, 3], 1), bounds=[-1, 1], file=f)
'''
