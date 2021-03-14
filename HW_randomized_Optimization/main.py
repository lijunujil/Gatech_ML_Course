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
from Config import *

sys.modules['sklearn.externals.six'] = six
import mlrose
import matplotlib.pyplot as plt

from RandomizedOptimization import RandomizedOptimization, find_best
from NeuralNetwork import NeuralNetwork



if __name__ == "__main__":
    # ro = RandomizedOptimization({"one_max": 500, "four_peak": 200, "max_k_color": 200, "queens": 80, "travelling_sales": 50})
    # ro = RandomizedOptimization({"one_max": 5, "four_peak": 5, "max_k_color": 5, "queens": 5, "travelling_sales": 50})
    # ro.run_experiment()
    find_best({"travelling_sales": 100}, 2)
    find_best({"four_peak": 100}, 2)
    find_best({"travelling_sales": 100}, 2)

    nn = NeuralNetwork()
    nn.run_experiment()
