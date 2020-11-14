import random
from random import seed
import numpy as np
import ANN as ann

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
    NN = ann.new_network()
    result =

    return


def PSO(swarm_size, velocity, p_best, i_best, g_best, max_iter, dimensions, bounds):
    g_best_value = float('inf')
    g_best_pos = np.array([float('inf'), float('inf')])
    arr = [Particle(np.array([random.uniform(bounds[0],bounds[1]) for i in range(dimensions)]), np.array(random.uniform(0, 1)), i) for i in range(swarm_size)]
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

        if abs(g_best_value - 1) < 1:

        iter += 1


PSO(swarm_size=10, velocity=1, p_best=1, i_best=1, g_best=1, max_iter=5, dimensions=2, bounds=[-1, 1])





