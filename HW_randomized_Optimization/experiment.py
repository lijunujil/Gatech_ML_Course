from Config import *
import six
import sys
import pandas as pd
import numpy as np
import pickle
import time
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from NeuralNetwork import NeuralNetwork
from RandomizedOptimization import find_best, RandomizedOptimization

def traveling_sales_max(state, coords):

    if len(state) != len(coords):
        raise Exception("""state must have the same length as coords.""")

    if not len(state) == len(set(state)):
        raise Exception("""Each node must appear exactly once in state.""")

    if min(state) < 0:
        raise Exception("""All elements of state must be non-negative"""
                        + """ integers.""")

    if max(state) >= len(state):
        raise Exception("""All elements of state must be less than"""
                        + """ len(state).""")

    fitness = 0

    # Calculate length of each leg of journey
    for i in range(len(state) - 1):
        node1 = state[i]
        node2 = state[i + 1]

        fitness += np.linalg.norm(np.array(coords[node1]) - np.array(coords[node2]))

    # Calculate length of final leg
    node1 = state[-1]
    node2 = state[0]

    fitness += np.linalg.norm(np.array(coords[node1]) - np.array(coords[node2]))

    return -fitness

# sys.modules['sklearn.externals.six'] = six
# import mlrose
# import matplotlib.pyplot as plt
#
# coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]
#
# # Initialize fitness function object using coords_list
# fitness_coords = mlrose.TravellingSales(coords = coords_list)
#
# problem_fit = mlrose.TSPOpt(length = 8, fitness_fn = fitness_coords, maximize=True)
# mimic = mlrose.mimic(problem_fit, curve=True)
# print(mimic)
#
# fitness = mlrose.CustomFitness(traveling_sales_max, problem_type='tsp', coords=coords_list)
# problem_fit = mlrose.TSPOpt(length = 8, fitness_fn = fitness, maximize=True)
# mimic = mlrose.mimic(problem_fit, curve=True)
# print(mimic)
#
# problem_fit = mlrose.TSPOpt(length = 8, fitness_fn = fitness_coords, maximize=False)
# mimic = mlrose.mimic(problem_fit, curve=True)
# print(mimic)

if __name__ == "__main__":
    # ro = RandomizedOptimization({"one_max": 500, "four_peak": 200, "max_k_color": 200, "queens": 80, "travelling_sales": 50})
    # ro = RandomizedOptimization({"one_max": 5, "four_peak": 5, "max_k_color": 5, "queens": 5, "travelling_sales": 50})
    # ro.run_experiment()

    # nn = NeuralNetwork()
    # nn.run_experiment()

    ro = RandomizedOptimization({"one_max": 20})
    ro.one_max()