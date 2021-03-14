from RandomizedOptimization import RandomizedOptimization

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


class ParameterSearch(RandomizedOptimization):
    def genetic_alg_search(self):
        result_list = []
        for pop_size in np.arange(2, 300, 10):
            for mutation_prob in np.arange(0.05, 0.8, 0.05):
        # for pop_size in np.arange(2, 300, 100):
        #     for mutation_prob in np.arange(0.05, 0.8, 0.4):
                for problem_name in ["one_max", "four_peak", "max_k_color", "queens", "travelling_sales"]:
                    fitness, problem = getattr(self, problem_name)
                    start = time.time()
                    state, score, curve = mlrose.genetic_alg(problem=problem, curve=True, pop_size=int(pop_size), mutation_prob=float(mutation_prob))
                    end = time.time()
                    print(problem_name, state, score, curve)
                    result_dict = {"problem": problem_name, "state": state, "score":score, "curve": curve, "time": end-start, "pop_size": pop_size, "mutation_prob": mutation_prob}
                    result_list.append(result_dict)

                    pd.DataFrame(result_list).to_excel("genetic_alg_search.xlsx")

    def simulated_annealing_search(self):
        result_list = []
        for decay in [mlrose.GeomDecay(), mlrose.ArithDecay(), mlrose.ExpDecay()]:
            for problem_name in ["one_max", "four_peak", "max_k_color", "queens", "travelling_sales"]:
                fitness, problem = getattr(self, problem_name)
                start = time.time()
                state, score, curve = mlrose.simulated_annealing(problem=problem, curve=True, schedule=decay)
                end = time.time()
                print(problem_name, state, score, curve)
                result_dict = {"problem": problem_name, "state": state, "score":score, "curve": curve, "time": end-start, "decay_name": decay}
                result_list.append(result_dict)

                pd.DataFrame(result_list).to_excel("simulated_annealing_search.xlsx")

    def mimic_search(self):
        result_list = []
        for pop_size in np.arange(2, 300, 10):
            for keep_pct in np.arange(0.05, 0.8, 0.05):
        # for pop_size in np.arange(2, 300, 100):
        #     for keep_pct in np.arange(0.05, 0.8, 0.4):
                for problem_name in ["one_max", "four_peak", "max_k_color", "queens", "travelling_sales"]:
                    fitness, problem = getattr(self, problem_name)
                    start = time.time()
                    state, score, curve = mlrose.mimic(problem=problem, curve=True, pop_size=int(pop_size), keep_pct=float(keep_pct))
                    end = time.time()
                    print(problem_name, state, score, curve)
                    result_dict = {"problem": problem_name, "state": state, "score":score, "curve": curve, "time": end-start, "pop_size": pop_size, "keep_pct": keep_pct}
                    result_list.append(result_dict)

                    pd.DataFrame(result_list).to_excel("mimic_search.xlsx")

    @property
    def one_max(self):
        fitness = mlrose.OneMax()
        problem = mlrose.DiscreteOpt(length=self.complexities["one_max"], fitness_fn=fitness, maximize=True, max_val=2)

        return fitness, problem

    @property
    def four_peak(self):
        fitness = mlrose.FourPeaks(t_pct=0.15)
        problem = mlrose.DiscreteOpt(length=self.complexities["four_peak"], fitness_fn=fitness, maximize=True, max_val=2)

        return fitness, problem

    @property
    def max_k_color(self):
        def random_edge_generator(length=10):
            random_edges = []
            for i in range(length):
                n_of_edge = np.random.randint(0, length-i)
                rand_int_list = np.random.choice(range(i+1, length), n_of_edge, replace=False)
                random_edges.extend([(i, node) for node in rand_int_list])
            return random_edges

        # edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        edges = random_edge_generator(self.complexities["max_k_color"])
        # edges = [(0, 1), (0, 3), (1, 3), (1, 2), (2, 3)]
        print("random edges", edges)
        # fitness = mlrose.MaxKColor(edges)
        fitness = mlrose.CustomFitness(self.k_color_max, problem_type='discrete', edges=edges)
        problem = mlrose.DiscreteOpt(length=self.complexities["max_k_color"], fitness_fn=fitness, maximize=True)

        return fitness, problem

    @property
    def queens(self):
        fitness = mlrose.CustomFitness(self.queens_max)
        problem = mlrose.DiscreteOpt(length=self.complexities["queens"], fitness_fn=fitness, maximize=True, max_val=self.complexities["queens"])

        return fitness, problem

    @property
    def travelling_sales(self):
        n = self.complexities["travelling_sales"]
        coords_list = random.sample([(x, y) for x in range(n*2) for y in range(n*2)], n)
        # fitness = mlrose.TravellingSales(coords=coords_list)
        fitness = mlrose.CustomFitness(self.traveling_sales_max, problem_type='tsp', coords=coords_list)
        problem = mlrose.TSPOpt(length=n, fitness_fn=fitness, maximize=False)

        return fitness, problem

if __name__ == "__main__":
    # ps = ParameterSearch({"one_max": 200, "four_peak": 100, "max_k_color": 100, "queens": 50, "travelling_sales": 50})
    ps = ParameterSearch({"one_max": 500, "four_peak": 200, "max_k_color": 200, "queens": 80, "travelling_sales": 50})
    # ps = ParameterSearch({"one_max": 5, "four_peak": 5, "max_k_color": 5, "queens": 5, "travelling_sales": 5})
    ps.genetic_alg_search()
    ps.simulated_annealing_search()
    ps.mimic_search()
