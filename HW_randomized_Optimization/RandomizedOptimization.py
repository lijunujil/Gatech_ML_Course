from Config import *
import pandas as pd
import numpy as np
import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose
import matplotlib.pyplot as plt


class RandomizedOptimization:
    def __init__(self, complexities):
        self.complexities = complexities
        self.algorithm_list = ["random_hill_climb", "simulated_annealing", "genetic_alg", "mimic"]
        self.config = Config()

    def run_experiment(self):
        # self.one_max()
        # self.four_peak()
        # self.max_k_color()
        # self.queens()
        self.travelling_sales()

    def one_max(self):
        fitness = mlrose.OneMax()
        problem = mlrose.DiscreteOpt(length=self.complexities["one_max"], fitness_fn=fitness, maximize=True, max_val=2)

        results = self.__construct_algorithm(problem=problem, init_state=None, discreption="one_max")
        return results

    def four_peak(self):
        fitness = mlrose.FourPeaks(t_pct=0.15)
        problem = mlrose.DiscreteOpt(length=self.complexities["four_peak"], fitness_fn=fitness, maximize=True,
                                     max_val=2)
        results = self.__construct_algorithm(problem=problem, init_state=None, discreption="four_peak")

        return results

    def max_k_color(self):
        def random_edge_generator(length=10):
            random_edges = []
            for i in range(length):
                n_of_edge = np.random.randint(0, length - i)
                rand_int_list = np.random.choice(range(i + 1, length), n_of_edge, replace=False)
                random_edges.extend([(i, node) for node in rand_int_list])
            return random_edges

        # edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        edges = random_edge_generator(self.complexities["max_k_color"])
        # edges = [(0, 1), (0, 3), (1, 3), (1, 2), (2, 3)]
        print("random edges", edges)
        # fitness = mlrose.MaxKColor(edges)
        fitness = mlrose.CustomFitness(self.k_color_max, problem_type='discrete', edges=edges)
        problem = mlrose.DiscreteOpt(length=self.complexities["max_k_color"], fitness_fn=fitness, maximize=True)

        results = self.__construct_algorithm(problem=problem, init_state=None, discreption="max_k_color")

        return results

    def queens(self):
        fitness = mlrose.CustomFitness(self.queens_max)
        problem = mlrose.DiscreteOpt(length=self.complexities["queens"], fitness_fn=fitness, maximize=True,
                                     max_val=self.complexities["queens"])

        results = self.__construct_algorithm(problem=problem, init_state=None, discreption="queens")

        return results

    def travelling_sales(self):
        n = self.complexities["travelling_sales"]
        coords_list = random.sample([(x, y) for x in range(n * 2) for y in range(n * 2)], n)
        # fitness = mlrose.TravellingSales(coords=coords_list)
        # problem = mlrose.TSPOpt(length=n, fitness_fn=fitness, maximize=False)

        fitness = mlrose.CustomFitness(self.traveling_sales_max, problem_type='tsp', coords=coords_list)
        problem = mlrose.TSPOpt(length=n, fitness_fn=fitness, maximize=True)

        results = self.__construct_algorithm(problem=problem, init_state=None, discreption="travelling_sales")

        return results

    def traveling_sales_max(self, state, coords):

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

    def k_color_max(self, state, edges):
        fitness = 0

        for i in range(len(edges)):
            # Check for adjacent nodes of the different color, we want to maximize this
            if state[edges[i][0]] != state[edges[i][1]]:
                fitness += 1

        return fitness

    def queens_max(self, state):
        # Define alternative N-Queens fitness function for maximization problem

        # Initialize counter
        fitness_cnt = 0

        # For all pairs of queens
        for i in range(len(state) - 1):
            for j in range(i + 1, len(state)):

                # Check for horizontal, diagonal-up and diagonal-down attacks
                if (state[j] != state[i]) \
                        and (state[j] != state[i] + (j - i)) \
                        and (state[j] != state[i] - (j - i)):
                    # If no attacks, then increment counter
                    fitness_cnt += 1

        return fitness_cnt

    def __construct_algorithm(self, problem, init_state, discreption):
        start = time.time()
        self.random_hill_climb = mlrose.random_hill_climb(problem,
                                                          max_attempts=self.config.max_attempts,
                                                          max_iters=self.config.max_iters,
                                                          restarts=self.config.restarts,
                                                          init_state=init_state,
                                                          curve=self.config.curve,
                                                          random_state=self.config.random_state)
        time1 = time.time()
        self.simulated_annealing = mlrose.simulated_annealing(problem,
                                                              schedule=self.config.schedule,
                                                              max_attempts=self.config.max_attempts,
                                                              max_iters=self.config.max_iters,
                                                              init_state=init_state,
                                                              curve=self.config.curve,
                                                              random_state=self.config.random_state)
        time2 = time.time()
        self.genetic_alg = mlrose.genetic_alg(problem,
                                              pop_size=self.config.pop_size,
                                              mutation_prob=self.config.mutation_prob,
                                              max_attempts=self.config.max_attempts,
                                              max_iters=self.config.max_iters,
                                              curve=self.config.curve,
                                              random_state=self.config.random_state)
        time3 = time.time()
        self.mimic = mlrose.mimic(problem,
                                  pop_size=self.config.pop_size,
                                  keep_pct=self.config.keep_pct,
                                  max_attempts=self.config.max_attempts,
                                  max_iters=self.config.max_iters,
                                  curve=self.config.curve,
                                  random_state=self.config.random_state,
                                  fast_mimic=self.config.fast_mimic)
        time4 = time.time()

        implementation_list = [self.random_hill_climb, self.simulated_annealing, self.genetic_alg, self.mimic]
        time_dict = {"random_hill_climb": time1 - start, "simulated_annealing": time2 - time1,
                     "genetic_alg": time3 - time2, "mimic": time4 - time3}
        result_dict = dict(zip(self.algorithm_list, implementation_list))
        for name, algo_results in result_dict.items():
            print(discreption, name, algo_results)
        print(time_dict)

        filehandler = open(f"Graph//{discreption}", "wb")
        pickle.dump(result_dict, filehandler)
        filehandler.close()

        plot_dict_data(tittle=f"{discreption} fitness curve", data_dict=result_dict,
                       complexity=self.complexities[discreption])

        return result_dict, time_dict


def find_best(complexity_dict, interval):
    score_dict = {"random_hill_climb": [], "simulated_annealing": [],"genetic_alg": [], "mimic": []}
    time_dict = {"random_hill_climb": [], "simulated_annealing": [],"genetic_alg": [], "mimic": []}
    for problem_name, val in complexity_dict.items():
        for n in np.arange(5, val, interval):
            ro = RandomizedOptimization({problem_name: int(n)})
            problem_to_run = getattr(ro, problem_name)
            result, performance_result = problem_to_run()
            # print(result)

            for algo in ["random_hill_climb", "simulated_annealing","genetic_alg", "mimic"]:
                score_dict[algo].append(result[algo][1])
                time_dict[algo].append(performance_result[algo])


            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f"{problem_name} problem fitness plot", fontsize=20)
            axes[0].grid()
            axes[0].set_title(f"fitness score")
            axes[0].set_xlabel("Complexity")
            axes[0].set_ylabel("fitness score")

            axes[1].grid()
            axes[1].set_xlabel("Complexity")
            axes[1].set_ylabel("run times in seconds")
            axes[1].set_title("Scalability of the model")
            for algo in ["random_hill_climb", "simulated_annealing","genetic_alg", "mimic"]:

                axes[0].plot(np.arange(5, n+interval, interval), score_dict[algo], label=f"{algo}")
                axes[1].plot(np.arange(5, n+interval, interval), time_dict[algo], label=f"{algo}")

            axes[0].legend(loc="best")
            axes[1].legend(loc="best")

            path = r"C:\Users\Lijun\Box\Gatech\CS7641_ML\HW_randomized_Optimization\Graph\Best_plot"
            filehandler = open(f"{path}//{problem_name}", "wb")
            pickle.dump([score_dict, time_dict], filehandler)
            filehandler.close()

            fig.savefig(os.path.join(path, f"{problem_name}"))
            fig.clf()

        print(score_dict, time_dict)


if __name__ == "__main__":
    # ro = RandomizedOptimization({"one_max": 500, "four_peak": 200, "max_k_color": 200, "queens": 80, "travelling_sales": 50})
    # ro = RandomizedOptimization({"one_max": 5, "four_peak": 5, "max_k_color": 5, "queens": 5, "travelling_sales": 50})
    # ro.run_experiment()

    # for complexity in range(1, 10):
    #     ro = RandomizedOptimization({"one_max": complexity})
    #     print(ro.one_max())

    find_best({"travelling_sales": 100}, 2)
    # find_best({"one_max": 500, "four_peak": 500, "max_k_color": 300}, 30)
    # find_best({"travelling_sales": 100, "queens": 100}, 10)

