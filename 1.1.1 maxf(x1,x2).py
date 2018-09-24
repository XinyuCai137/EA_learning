# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 2018
@author: Vegelofe
Learned from Morvan Python
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


DNA_SIZE = 33            # DNA length
DNA_SIZE_X1 = 18
DNA_SIZE_X2 = 15
POP_SIZE = 100           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.01     # mutation probability
N_GENERATIONS = 1000
X1_MAX = 12.1
X1_MIN = -3
X1_RANGE = 262143
X2_MAX = 5.8
X2_MIN = 4.1
X2_RANGE = 32767
ACCURACY = 10000


def f(x) :          # define f(x1,x2)
    x1 = x[0]
    x2 = x[1]
    return 21.5 + x1 * np.sin(4 * np.pi * x1) + x2 * np.sin(20 * np.pi * x2)

def translateDNA(pop) :
    x1 = pop[:,0:18].dot(2 ** np.arange(DNA_SIZE_X1)[::-1]) / X1_RANGE * 15.1 - 3
    x2 = pop[:,18:33].dot(2 ** np.arange(DNA_SIZE_X2)[::-1]) / X2_RANGE * 1.7 + 4.1
    return (x1, x2)


def crossover(parent, pop) :
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]

# plt.ion()       # something about plotting

# fig = plt.figure()
# ax = Axes3D(fig)

# x1 = np.linspace(X1_MIN, X1_MAX, 100).transpose()
# x2 = np.linspace(X2_MIN, X2_MAX, 100).transpose()

# x1, x2 = np.meshgrid(x1, x2)

# z = 21.5 + x1 * np.sin(4 * np.pi * x1) + x2 * np.sin(20 * np.pi * x2)

# ax.plot_surface(x1, x2, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))


if __name__ == '__main__':

    pop = (np.random.randint(2, size=(POP_SIZE, DNA_SIZE)))
    M_value = []

    for _ in range(N_GENERATIONS) :
        F_value = f(translateDNA(pop))
        print("Most fitted DNA: ", pop[np.argmax(F_value), :], np.max(F_value))
        M_value.append(np.max(F_value))
        pop = select(pop, F_value)
        pop_copy = pop.copy()
        for parent in pop:
            child = crossover(parent, pop_copy)
            child = mutate(child)
            parent[:] = child       # parent is replaced by its child

    plt.plot(np.array(M_value))
    plt.show()