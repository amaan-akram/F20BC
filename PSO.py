import random
import math
from random import seed
import numpy as np
import ANN as ann

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



def fitness(particle):
    # this is the loss function
    x = particle.position
    fitness_position = 3 * (1 - x[0]) ** 2 * math.exp(-x[0] ** 2 - (x[1] + 1) ** 2) - 10 * ( x[0] / 5 - x[0] ** 3 - x[1] ** 5) * math.exp(-x[0] ** 2 - x[1] ** 2) - 1 / 3 * math.exp( -(x[0] + 1) ** 2 - x[1] ** 2);
    return fitness_position


def PSO(swarm_size, velocity, p_best, i_best, g_best, max_iter, dimensions, bounds):
    g_best_value = float('inf')
    g_best_pos = np.array([float('inf'), float('inf')])
    arr = [Particle(np.array([random.uniform(bounds[0], bounds[1]) for i in range(dimensions)]), np.array(random.uniform(0, 1)), i) for i in range(swarm_size)]
    for i in arr:
        i.particlePos()
    iter = 0
    while iter < max_iter:
        for particle in arr:
            particle_fitness = fitness(particle)
            if particle.pBest_value > particle_fitness:
                particle.pBest_value = particle_fitness
                particle.pBest = particle.position

            if g_best_value > particle_fitness:
                g_best_value = particle_fitness
                g_best_pos = particle.position


        iter += 1


network = ann.createNN(1, [3, 3, 3], 1, sigmoid)
result = ann.forward(network, [1])
print(result)
PSO(swarm_size=10, velocity=1, p_best=1, i_best=1, g_best=1, max_iter=5, dimensions=24, bounds=[-1, 1])





