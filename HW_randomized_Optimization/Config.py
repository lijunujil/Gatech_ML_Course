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

sys.modules['sklearn.externals.six'] = six
import mlrose
import matplotlib.pyplot as plt


# class Config:
#     def __init__(self):
#         # common attributes
#         self.max_attempts = 100
#         self.max_iters = 10000
#         self.restarts = 0
#         self.curve = True
#         self.random_state = 1
#
#         # for simulated_annealing
#         self.schedule = mlrose.ExpDecay()
#
#         # for genetic_alg
#         self.pop_size = 1000
#         self.mutation_prob = 0.15
#
#         # for MIMIC
#         self.keep_pct = 0.4
#         self.fast_mimic = False

class Config:
    def __init__(self):
        # common attributes
        self.max_attempts = 100
        self.max_iters = np.inf
        self.restarts = 0
        self.curve = True
        self.random_state = 1

        # for simulated_annealing
        self.schedule = mlrose.ExpDecay()

        # for genetic_alg
        self.pop_size = 1000
        self.mutation_prob = 0.05

        # for MIMIC
        self.keep_pct = 0.2
        self.fast_mimic = False


def plot_dict_data(tittle: str, data_dict: dict, complexity, xlabel="iterations", ylabel="best fitness value"):
    plt.figure(figsize=(8, 6))
    plt.title(tittle)

    for key, value in data_dict.items():
        plt.plot(value[2], label=key)

    plt.xlabel = xlabel
    plt.ylabel = ylabel
    plt.legend()
    plt.savefig(os.path.join("Graph", f"{tittle} {complexity}.png"))
    plt.close()