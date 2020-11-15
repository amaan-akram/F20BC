import random
import math
from random import seed
import numpy as np
import ANN as ann
import data_prep as dp

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

class Particle:
    def __init__(self, position, velocity, num):
        self.position = position
        self.pBest = self.position
        self.pBest_value = float('inf')
        self.velocity = velocity
        self.id = num

    def particlePos(self):
        print("Particle ", self.id, " at position : ", self.position, " and PB is ", self.pBest,
              " with velocity ", self.velocity)



def fitness(particle, inp, pred):
    x = particle.position
    print(x)
    # update weights
    # this is the loss function
    values = []
    for i in inp:
        result = ann.forward(network, [i])
        values.append(result)

    # ANN results
    total = 0
    for i in range(len(values)):
        total += np.mean((values[i] - pred[i])**2)
    return total / len(values)


def PSO(swarm_size, velocity, p_best, i_best, g_best, max_iter, dimensions, bounds, file):
    inputs, exp = dp.prepare_data(file)
    g_best_value = float('inf')
    g_best_pos = np.array([float('inf'), float('inf')])
    arr = [Particle(np.array([random.uniform(bounds[0], bounds[1]) for i in range(dimensions)]), np.array(random.uniform(0, 1)), i) for i in range(swarm_size)]

    iter = 0
    while iter < max_iter:
        for particle in arr:
            particle_fitness = fitness(particle, inputs, exp)
            if particle.pBest_value > particle_fitness:
                particle.pBest_value = particle_fitness
                particle.pBest = particle.position

            if g_best_value > particle_fitness:
                g_best_value = particle_fitness
                g_best_pos = particle.position


        iter += 1


network = ann.createNN(1, [3, 3, 3], 1, sigmoid)

#PSO(swarm_size=10, velocity=1, p_best=1, i_best=1, g_best=1, max_iter=5, dimensions=24, bounds=[-1, 1], file=file)





