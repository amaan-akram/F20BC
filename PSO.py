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

    def newVelocity(self, vel_const, g_best, i_best, p_best, i):

        for j in range(i):
            r1 = random.random()
            r2 = random.random()
            new_Vel = vel_const * self.velocity + p_best * r1 * (self.pBest[j] - self.position[j]) + (i_best * r2 * (self.position[j] - g_best))

        return new_Vel


def fitness(particle, inp, pred):
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
        total += np.mean((values[i][0] - pred[i])**2)
    return total / len(values)






def PSO(swarm_size, vel_const, p_best, i_best, g_best, max_iter, dimensions, bounds, file):
    inputs, exp = dp.prepare_data(file)
    g_best_value = float('inf')
    g_best_pos = np.array([float('inf'), float('inf')])
    arr = [Particle([random.uniform(bounds[0], bounds[1]) for i in range(dimensions)], random.uniform(0, 1), i) for i in range(swarm_size)]

    iter = 0
    while iter < max_iter:

        for particle in arr:




            for i in range(len(particle.position)):
                newVel = particle.newVelocity(vel_const=vel_const, p_best=p_best, i_best=i_best, g_best=g_best,
                                              i=dimensions)
                particle.position[i] = particle.position[i] + newVel


            particle_fitness = fitness(particle, inputs, exp)

            if particle.pBest_value > particle_fitness:
                particle.pBest_value = particle_fitness
                particle.pBest = particle.position

            if g_best_value > particle_fitness:
                g_best_value = particle_fitness
                g_best_pos = particle.position

            print("Particle ", particle.id," : ",particle.pBest_value)
            print("Particle abc ", particle.id, " : ", particle.pBest[0])
            print("GolbalBest: ", g_best_value)

        iter += 1






network = ann.createNN(1, [3, 3, 3], 1, ann.hyperbolic_Tangent)

f = "Data/1in_cubic.txt"

PSO(swarm_size=10, vel_const=0.8, p_best=0.5, i_best=0.2, g_best=0.3, max_iter=100, dimensions=ann.dimensions_num(1, [3, 3, 3], 1), bounds=[-1, 1], file=f)





