import random
import ANN as ann
import data_prep as dp
import matplotlib.pyplot as plot


class Particle:
    def __init__(self, position, num):
        self.position = position
        self.velocity = [random.uniform(0, 1) for _ in range(len(self.position))]
        self.error = float("inf")
        self.best_position = self.position
        self.best_error = self.error
        self.informants = []
        self.num = num

    def bestInformant(self):
        best = float('inf')
        best_inf_pos = self.position
        for inf in self.informants:
            if best > inf.best_error:
                best_inf_pos = inf.best_position
        return best_inf_pos

    def calc_velocity(self, global_best_position, informant_position, max_vel, max_pb, max_ib, max_gb):
        for i in range(len(self.position)):
            w = random.uniform(0, max_vel)
            c1 = random.uniform(0, max_pb)
            c2 = random.uniform(0, max_ib)
            c3 = random.uniform(0, max_gb)
            new_vel = ((w * self.velocity[i]) + c1 * (self.best_position[i] - self.position[i]) + (
                        c2 * (informant_position[i] - self.position[i])) +
                       (c3 * (global_best_position[i] - self.position[i])))
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


def PSO(iterations, swarm_size, bounds, max_vel, max_pb, max_ib, max_gb):
    GLOBAL_BEST_POSITION = []
    GLOBAL_BEST_ERROR = float("inf")
    i = 0
    swarm = [Particle([random.uniform(bounds[0], bounds[1]) for k in range(dim)], num=j) for j in range(swarm_size)]
    while i < iterations:
        print("iter", i)
        # get informants
        for particle in swarm:
            particle.informants = []

        rand = []
        for j in range(0, (len(swarm) // 2)):
            n = random.randint(0, len(swarm) - 1)
            rand.append(n)

        for particle in swarm:

            for j in rand:
                particle.informants.append(swarm[j])

        for particle in swarm:

            particle.fitness(inp, output)
            if particle.best_error > particle.error:
                particle.best_error = particle.error
                particle.best_position = particle.position

            if particle.error < GLOBAL_BEST_ERROR:
                GLOBAL_BEST_ERROR = particle.error
                GLOBAL_BEST_POSITION = particle.position

        print("ITER : ", i, " Global best error = ", GLOBAL_BEST_ERROR)

        for particle in swarm:
            particle.calc_velocity(GLOBAL_BEST_POSITION, particle.bestInformant(), max_vel, max_pb, max_ib, max_gb)
            particle.update_position()

        i += 1

    return GLOBAL_BEST_POSITION
    # out1 = original   out2 = result


def plotGraph(inp1, out1, out2):
    plot.plot(inp1, out1)
    plot.plot(inp1, out2)
    plot.show()


network = ann.createNN(1, [3], 1, ann.hyperbolic_Tangent)
dim = ann.dimensions_num(1, [3], 1)  # MAKE SURE TO CHANGE THIS
inp, output = dp.prepare_data("Data/1in_cubic.txt")

BEST_OVERALL = PSO(iterations=50, swarm_size=100, bounds=[-5, 5], max_vel=1, max_pb=1.5, max_ib=1.2, max_gb=0.1)
predicted_values = []
print(BEST_OVERALL)
print(len(BEST_OVERALL))
ann.updateAllWeights(network, BEST_OVERALL)

for i in inp:
    predicted_values.append(ann.forward(network, [i]))
plotGraph(inp, output, predicted_values)

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
