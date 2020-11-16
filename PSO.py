import random
import math
from random import seed
import numpy as np
import ANN as ann
import data_prep as dp
import plots


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


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

PSO(swarm_size=50, vel_const=0.8, p_best=0.5, i_best=0.2, g_best=0.3, max_iter=50,
    dimensions=ann.dimensions_num(1, [3, 3, 3], 1), bounds=[-3, 3], file=f)
