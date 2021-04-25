# import mdptoolbox
# import mdptoolbox.example

import hiive.mdptoolbox.mdp
import hiive.mdptoolbox.example
# import ModifiedMDP.example
# import ModifiedMDP.mdp
from gym.envs.toy_text.frozen_lake import generate_random_map
import gym
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import itertools
import random
import pandas as pd
from collections import defaultdict
np.set_printoptions(threshold=sys.maxsize)

random.seed(0)
np.random.seed(0)


def main():
    # forest_mdp()
    # frozen_lake_mdp()

    # # Runtime and number of iter
    mdp_experiments("forest")
    mdp_experiments("frozen_lake")

    plot_learned_policy(n_state=25, gamma=0.95, is_slippery=False)
    plot_learned_policy(n_state=25, gamma=0.95)
    plot_learned_policy(n_state=900, gamma=0.95)

    # forest reward experiments
    forest_reward_experiments(n_state=5, r1=10000)
    forest_reward_experiments(n_state=50, r1=10000)

    # frozen_lake_experiments(n_state=25, gamma=0.95, is_slippery=False)
    frozen_lake_experiments(n_state=25, gamma=0.95)
    frozen_lake_experiments(n_state=900, gamma=0.95)


def plot_learned_policy(n_state=16, gamma=0.95, is_slippery=True):
    transition_p, reward, random_map = frozen_lake_mdp_helper(grid_size=int(np.sqrt(n_state)), p=0.9, is_slippery=is_slippery)
    hole_list = [n for n, x in enumerate(list(itertools.chain.from_iterable([i for i in random_map]))) if x == 'H']
    hole_list.extend([0, n_state-1])
    def iter_callback(x, y, z):
        return True if x in hole_list else False
    learner_dict = {"Policy Iteration": policy_iteration(**{"transitions": transition_p, "reward": reward, "gamma": gamma, "eval_type": 1}),
                    "Value Iteration": value_iteration(**{"transitions": transition_p, "reward": reward, "gamma": gamma}),
                    "Q Learning": q_learning(**{"transitions": transition_p, "reward": reward, "gamma": gamma, "n_iter":100000, "iter_callback": iter_callback})}

    # pi = policy_iteration(**{"transitions": transition_p, "reward": reward, "gamma": gamma, "eval_type": 1})
    # vi = value_iteration(**{"transitions": transition_p, "reward": reward, "gamma": gamma})
    # ql = q_learning(**{"transitions": transition_p, "reward": reward, "gamma": gamma})
    for learner_name, learner in learner_dict.items():

        policy = np.array(learner.policy)
        policy = np.array([val if idx not in hole_list else 4 for idx, val in enumerate(policy)])
        policy_reshaped = list(policy.reshape((int(np.sqrt(n_state)), int(np.sqrt(n_state)))))

        # LEFT = 0, u"\u2190", DOWN = 1, u"\u2193", RIGHT = 2, u"\u2192", UP = 3, u"\u2191"
        arrow_map = {0: u"\u2190", 1: u"\u2193", 2: u"\u2192", 3: u"\u2191", 4: "", "Start": "Start", "Goal": "Goal"}
        value_map = {"S": 1, "F": 0, "G": 1, "H": -1}

        random_map = [[j for j in i] for i in random_map]
        random_map_int = [[value_map[j] for j in i] for i in random_map]
        action_map = [[arrow_map[j] for j in i] for i in policy_reshaped]
        action_map[0][0] = "Start"
        action_map[-1][-1] = "Goal"
        print(learner_name, policy)
        game_type = "Stochastic" if is_slippery else "Deterministic"

        plt.figure(figsize=(13, 6.5))
        sns.heatmap(random_map_int, annot=action_map, cmap='coolwarm', fmt='')
        plt.title(f"Policy for {learner_name} in {game_type} frozen lake game, with n_state = {n_state}")
        plt.savefig(os.path.join("Graph", f"{learner_name} in {game_type} frozen lake game, with n_state = {n_state}"))
        plt.close()


def plot_graph(x, y1, y2, x_name, y_name, tittle, sub1, sub2):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.suptitle(tittle, fontsize=20)
    axes[0].grid()
    axes[0].set_title(sub1)
    # axes[0].set_ylim(0.5, 1)
    axes[0].set_xlabel(x_name)
    axes[0].set_ylabel(y_name)
    axes[0].plot(x, y1, 'o-', color="g", label=y_name)
    axes[0].legend(loc="best")

    axes[1].grid()
    axes[1].set_title(sub2)
    axes[1].set_xlabel(x_name)
    axes[1].set_ylabel(y_name)
    axes[1].plot(x, y2, 'o-', color="g", label=y_name)
    axes[1].legend(loc="best")

    # fig.tight_layout()
    fig.savefig(os.path.join("Graph", tittle))
    plt.close()


def mdp_experiments(problem="forest"):

    # How did the number of states affect things, if at all?
    pi_runtime_list, vi_runtime_list = [], []
    pi_iter_list, vi_iter_list = [], []
    n_state_range = range(100, 4900, 100)
    for n_state in n_state_range:
        print("n_state=", n_state)
        if problem == "forest":
            transition_p, reward = hiive.mdptoolbox.example.forest(S=n_state, r1=4, r2=2, p=0.1, is_sparse=False)
        else:
            transition_p, reward, _ = frozen_lake_mdp_helper(grid_size=int(np.sqrt(n_state)), p=0.8)

        pi = policy_iteration(**{"transitions": transition_p, "reward": reward, "gamma": 0.9, "eval_type": 1})
        vi = value_iteration(**{"transitions": transition_p, "reward": reward, "gamma": 0.9})

        pi_runtime_list.append(pi.time)
        pi_iter_list.append(pi.iter)
        vi_runtime_list.append(vi.time)
        vi_iter_list.append(vi.iter)

    plot_graph(x=n_state_range, y1=pi_runtime_list, y2=vi_runtime_list,
               x_name="Number of state", y_name="Running time",
               tittle=f"Running time with different number of state for {problem} MDP",
               sub1="Policy iteration", sub2="Value iteration")
    plot_graph(x=n_state_range, y1=pi_iter_list, y2=vi_iter_list,
               x_name="Number of state", y_name="Number of iter",
               tittle=f"Number of iter with different number of state for {problem} MDP",
               sub1="Policy iteration", sub2="Value_iteration")

    # Learning rate
    pi_runtime_list, vi_runtime_list = [], []
    pi_iter_list, vi_iter_list = [], []
    n_state = 1600
    gamma_range = np.arange(0.01, 1., 0.01)
    for gamma in gamma_range:
        print("gamma=", gamma)
        if problem == "forest":
            transition_p, reward = hiive.mdptoolbox.example.forest(S=n_state, r1=4, r2=2, p=0.1, is_sparse=False)
        else:
            # int(np.sqrt(n_state)) because this is the n for row or col in the game matrix
            transition_p, reward, _ = frozen_lake_mdp_helper(grid_size=int(np.sqrt(n_state)), p=0.8)
        pi = policy_iteration(**{"transitions": transition_p, "reward": reward, "gamma": gamma, "eval_type": 1})
        vi = value_iteration(**{"transitions": transition_p, "reward": reward, "gamma": gamma})

        pi_runtime_list.append(pi.time)
        pi_iter_list.append(pi.iter)
        vi_runtime_list.append(vi.time)
        vi_iter_list.append(vi.iter)

    plot_graph(x=gamma_range, y1=pi_runtime_list, y2=vi_runtime_list,
               x_name="Discount rate", y_name="Running time",
               tittle=f"Running time with different discount rate, state={n_state}, for {problem} MDP",
               sub1="policy_iteration", sub2="value_iteration")
    plot_graph(x=gamma_range, y1=pi_iter_list, y2=vi_iter_list,
               x_name="Discount rate", y_name="Number of iter",
               tittle=f"number of iter with different discount rate, state={n_state}, for {problem} MDP",
               sub1="Policy iteration", sub2="Value_iteration")


def forest_mdp():
    # different discount factor as a hyper-parameters?
    # discount = 0.9

    # params_list = [{"discount": 0.9, "n_state": 100}, {"discount": 0.9, "n_state": 10000}]
    params_list = [{"discount": 0.9, "n_state": 4}, {"discount": 0.9, "n_state": 200}]

    for params in params_list:
        transition_p, reward = hiive.mdptoolbox.example.forest(S=params["n_state"], r1=4, r2=2, p=0.1, is_sparse=False)

        pi = policy_iteration(**{"transitions": transition_p, "reward": reward, "gamma": params["discount"]})
        vi = value_iteration(**{"transitions": transition_p, "reward": reward, "gamma": params["discount"]})
        ql = q_learning(**{"transitions": transition_p, "reward": reward, "gamma": params["discount"]})

        print("PI time, iter, policy, V", pi.time, "\n", pi.iter, "\n", pi.policy, "\n", pi.V, "\n")
        print("VI time, iter, policy, V", vi.time, "\n", vi.iter, "\n", vi.policy, "\n", vi.V, "\n")
        print("QL Q, policy, V", ql.Q, "\n", ql.policy, "\n", ql.V, "\n")


def frozen_lake_mdp():

    params_list = [{"discount": 0.9, "grid_size": 4, "p": 0.8}, {"discount": 0.9, "grid_size": 8, "p": 0.8}]

    for params in params_list:
        transition_p, reward, _ = frozen_lake_mdp_helper(params["grid_size"], params["p"])

        pi = policy_iteration(**{"transitions": transition_p, "reward": reward, "gamma": params["discount"]})
        vi = value_iteration(**{"transitions": transition_p, "reward": reward, "gamma": params["discount"]})
        ql = q_learning(**{"transitions": transition_p, "reward": reward, "gamma": params["discount"]})

        # print()


def frozen_lake_mdp_helper(grid_size=4, p=0.8, is_slippery=True):
    """
    (*) dictionary of lists, where
    P[s][a] == [(probability, next state, reward, done), ...]
    (**) list or array of length nS
    """
    n_action = 4

    random_map = generate_random_map(size=grid_size, p=p)
    env = gym.make("FrozenLake-v0", desc=random_map, is_slippery=is_slippery)
    env.reset()
    env.render()
    open_ai_p = env.P
    # print(env.P)

    transition_p = np.zeros((n_action, grid_size**2, grid_size**2))
    reward = np.zeros((n_action, grid_size**2, grid_size**2))

    for state, state_dict in open_ai_p.items():
        for action, prob_tuple_list in state_dict.items():
            for prob_tuple in prob_tuple_list:
                probability, next_state, r, done = prob_tuple

                transition_p[action][state][next_state] += probability
                reward[action][state][next_state] = r*100 - 1
    #             # print(r)
    #             if probability != 0:
    #                 print("Found", state, action, probability, next_state, r, done)
    # print(transition_p)
    # # print(reward)

    return transition_p, reward, random_map


# def frozen_lake_mdp_helper(grid_size=4, p=0.8):
#     """
#     (*) dictionary of lists, where
#     P[s][a] == [(probability, next state, reward, done), ...]
#     (**) list or array of length nS
#     """
#     n_action = 4
#
#     random_map = generate_random_map(size=grid_size, p=p)
#     hole_list = [n for n,x in enumerate(list(itertools.chain.from_iterable([i for i in random_map]))) if x=='H']
#     env = gym.make("FrozenLake-v0", desc=random_map)
#     env.reset()
#     env.render()
#     open_ai_p = env.P
#     # print(env.P)
#
#     transition_p = np.zeros((n_action, grid_size**2, grid_size**2))
#     reward = np.zeros((n_action, grid_size**2, grid_size**2))
#
#     for state, state_dict in open_ai_p.items():
#         for action, prob_tuple_list in state_dict.items():
#             for prob_tuple in prob_tuple_list:
#                 probability, next_state, r, done = prob_tuple
#
#                 if state in hole_list:
#                     transition_p[action][state][0] = 1
#                     reward[action][state][0] = -10
#                 else:
#                     transition_p[action][state][next_state] += probability
#                     reward[action][state][next_state] = r*100 - 1
#     #             # print(r)
#     #             if probability != 0:
#     #                 print("Found", state, action, probability, next_state, r, done)
#     # print(transition_p)
#     # # print(reward)
#
#     return transition_p, reward, random_map

# def forest_reward_experiments(n_state=10):
#
#     # params_list = [{"discount": 0.9, "n_state": 100}, {"discount": 0.9, "n_state": 10000}]
#     # discount_list = np.arange(0.1, 1, 0.1)
#     training_iteration_list = np.arange(10000, 100000, 10000)
#     pi_reward_list = []
#     vi_reward_list = []
#     ql_reward_list = []
#     gamma = 0.9
#
#     for max_iter in training_iteration_list:
#         transition_p, reward = hiive.mdptoolbox.example.forest(S=n_state, r1=40, r2=200, p=0, is_sparse=False)
#
#         pi = policy_iteration(**{"transitions": transition_p, "reward": reward, "gamma": gamma, "max_iter": max_iter})
#         vi = value_iteration(**{"transitions": transition_p, "reward": reward, "gamma": gamma, "max_iter": max_iter})
#         ql = q_learning(**{"transitions": transition_p, "reward": reward, "gamma": gamma, "n_iter": max_iter})
#
#         # print("PI time, iter, policy, V", pi.time, "\n", pi.iter, "\n", pi.policy, "\n", pi.V, "\n")
#         # print("VI time, iter, policy, V", vi.time, "\n", vi.iter, "\n", vi.policy, "\n", vi.V, "\n")
#
#         pi_reward_list.append(np.mean(policy_reward_cal(n_state, pi.policy, transition_p, reward, gamma, 10000)))
#         vi_reward_list.append(np.mean(policy_reward_cal(n_state, vi.policy, transition_p, reward, gamma, 10000)))
#         ql_reward_list.append(np.mean(policy_reward_cal(n_state, ql.policy, transition_p, reward, gamma, 10000)))
#
#     pi_table = pd.DataFrame(np.array(pi_reward_list).reshape(1, len(pi_reward_list)),
#                             columns=list(training_iteration_list), index=['Policy Iteration'])
#     vi_table = pd.DataFrame(np.array(vi_reward_list).reshape(1, len(vi_reward_list)),
#                             columns=list(training_iteration_list), index=['Value Iteration'])
#     ql_table = pd.DataFrame(np.array(ql_reward_list).reshape(1, len(ql_reward_list)),
#                             columns=list(training_iteration_list), index=['Reinforcement Learning'])
#
#     new_df = pd.concat([pi_table, vi_table, ql_table])
#     new_df.to_excel(f"training_iteration_list_experiment_forest——n-state={n_state}.xlsx")

def frozen_lake_experiments(n_state=16, gamma=0.95, max_iter=10000, is_slippery=True):

    result_dict = defaultdict(lambda: defaultdict(str))

    transition_p, reward, random_map = frozen_lake_mdp_helper(grid_size=int(np.sqrt(n_state)), p=0.9, is_slippery=is_slippery)
    absorbing_list = [n for n, x in enumerate(list(itertools.chain.from_iterable([i for i in random_map]))) if x == 'H']
    absorbing_list.extend([n_state - 1])

    pi = policy_iteration(**{"transitions": transition_p, "reward": reward, "gamma": gamma, "max_iter": max_iter})
    vi = value_iteration(**{"transitions": transition_p, "reward": reward, "gamma": gamma, "max_iter": max_iter})

    pi_reward = np.mean(frozen_lake_policy_reward_cal(absorbing_list, pi.policy, transition_p, reward, gamma, 10000))
    vi_reward = np.mean(frozen_lake_policy_reward_cal(absorbing_list, vi.policy, transition_p, reward, gamma, 10000))

    result_dict["Policy Iteration"]["Number of Iter"] = pi.iter
    result_dict["Policy Iteration"]["Time"] = pi.time
    result_dict["Policy Iteration"]["Expected Reward"] = str(pi_reward)
    result_dict["Policy Iteration"]["Policy"] = str(pi.policy)
    result_dict["Policy Iteration"]["Same Policy?"] = str(str(pi.policy) == str(vi.policy))

    result_dict["Value Iteration"]["Number of Iter"] = vi.iter
    result_dict["Value Iteration"]["Time"] = vi.time
    result_dict["Value Iteration"]["Expected Reward"] = str(vi_reward)
    result_dict["Value Iteration"]["Policy"] = str(vi.policy)

    # # Now compare with rl
    # alpha_list = np.arange(0.1, 1, 0.1)
    # hole_list = [n for n, x in enumerate(list(itertools.chain.from_iterable([i for i in random_map]))) if x == 'H']
    # hole_list.extend([0, n_state-1])
    # def iter_callback(x, y, z):
    #     return True if x in hole_list else False
    #
    # for alpha in alpha_list:
    #     ql = q_learning(**{"transitions": transition_p, "reward": reward, "gamma": gamma, "n_iter": max_iter, "alpha": alpha ,"iter_callback": iter_callback})
    #     ql_reward = np.mean(frozen_lake_policy_reward_cal(absorbing_list, ql.policy, transition_p, reward, gamma, 10000))
    #
    #     # result_dict[f"{alpha}"]["Number of Iter"] = vi.iter
    #     # result_dict[f"{alpha}"]["Time"] = vi.time
    #     result_dict[f"{alpha}"]["Expected Reward"] = str(ql_reward)
    #     result_dict[f"{alpha}"]["Policy"] = str(ql.policy)
    #
    #
    # # Now compare with rl
    # max_iter_list = np.arange(10000, 1000000, 100000)
    # for max_iter in max_iter_list:
    #     ql = q_learning(**{"transitions": transition_p, "reward": reward, "gamma": gamma, "n_iter": max_iter, "iter_callback": iter_callback})
    #     ql_reward = np.mean(frozen_lake_policy_reward_cal(absorbing_list, ql.policy, transition_p, reward, gamma, 10000))
    #
    #     result_dict[f"{max_iter}"]["Expected Reward"] = str(ql_reward)
    #     result_dict[f"{max_iter}"]["Policy"] = str(ql.policy)

    game_type = "Stochastic" if is_slippery else "Deterministic"
    pd.DataFrame(result_dict).to_excel(f"{game_type}_FrozenLake_Experiment_n-state={n_state}.xlsx")


def forest_reward_experiments(n_state=5, r1=10000, max_iter=100000):

    # params_list = [{"discount": 0.9, "n_state": 100}, {"discount": 0.9, "n_state": 10000}]
    # discount_list = np.arange(0.1, 1, 0.1)
    # training_iteration_list = np.arange(10000, 100000, 10000)
    result_dict = defaultdict(lambda: defaultdict(str))
    # ql_reward_list = []
    gamma = 0.95

    transition_p, reward = hiive.mdptoolbox.example.forest(S=n_state, r1=r1, r2=20, p=0.02, is_sparse=False)

    pi = policy_iteration(**{"transitions": transition_p, "reward": reward, "gamma": gamma, "max_iter": max_iter, "eval_type": 1})
    vi = value_iteration(**{"transitions": transition_p, "reward": reward, "gamma": gamma, "max_iter": max_iter})
    # ql = q_learning(**{"transitions": transition_p, "reward": reward, "gamma": gamma, "n_iter": max_iter})

    # print("PI time, iter, policy, V", pi.time, "\n", pi.iter, "\n", pi.policy, "\n", pi.V, "\n")
    # print("VI time, iter, policy, V", vi.time, "\n", vi.iter, "\n", vi.policy, "\n", vi.V, "\n")

    pi_reward = np.mean(forest_policy_reward_cal(n_state, pi.policy, transition_p, reward, gamma, 10000))
    vi_reward = np.mean(forest_policy_reward_cal(n_state, vi.policy, transition_p, reward, gamma, 10000))

    result_dict["Policy Iteration"]["Number of Iter"] = pi.iter
    result_dict["Policy Iteration"]["Time"] = pi.time
    result_dict["Policy Iteration"]["Expected Reward"] = str(pi_reward)
    result_dict["Policy Iteration"]["Policy"] = str(pi.policy)

    result_dict["Value Iteration"]["Number of Iter"] = vi.iter
    result_dict["Value Iteration"]["Time"] = vi.time
    result_dict["Value Iteration"]["Expected Reward"] = str(vi_reward)
    result_dict["Value Iteration"]["Policy"] = str(vi.policy)

    # Now compare with rl
    alpha_list = np.arange(0.1, 1, 0.1)
    for alpha in alpha_list:
        ql = q_learning(**{"transitions": transition_p, "reward": reward, "gamma": gamma, "n_iter": max_iter, "alpha": alpha})
        ql_reward = np.mean(forest_policy_reward_cal(n_state, ql.policy, transition_p, reward, gamma, 10000))

        # result_dict[f"{alpha}"]["Number of Iter"] = vi.iter
        # result_dict[f"{alpha}"]["Time"] = vi.time
        result_dict[f"alpha={alpha}"]["Expected Reward"] = str(ql_reward)
        result_dict[f"alpha={alpha}"]["Policy"] = str(ql.policy)
    #
    # # Now compare with rl
    # max_iter_list = np.arange(10000, 1000000, 100000)
    # for max_iter in max_iter_list:
    #     ql = q_learning(**{"transitions": transition_p, "reward": reward, "gamma": gamma, "n_iter": max_iter})
    #     ql_reward = np.mean(forest_policy_reward_cal(n_state, ql.policy, transition_p, reward, gamma, 10000))
    #
    #     result_dict[f"{max_iter}"]["Expected Reward"] = str(ql_reward)
    #     result_dict[f"{max_iter}"]["Policy"] = str(ql.policy)
    # Now compare with rl

    epsilon_list = np.arange(0.11, 1, 0.1)
    for epsilon in epsilon_list:
        ql = q_learning(**{"transitions": transition_p, "reward": reward, "gamma": gamma, "n_iter": max_iter, "epsilon": epsilon})
        ql_reward = np.mean(forest_policy_reward_cal(n_state, ql.policy, transition_p, reward, gamma, 10000))

        # result_dict[f"{alpha}"]["Number of Iter"] = vi.iter
        # result_dict[f"{alpha}"]["Time"] = vi.time
        result_dict[f"epsilon{epsilon}"]["Expected Reward"] = str(ql_reward)
        result_dict[f"epsilon{epsilon}"]["Policy"] = str(ql.policy)

    epsilon_min_list = np.arange(0.01, 0.1, 0.01)
    for epsilon_min in epsilon_min_list:
        ql = q_learning(**{"transitions": transition_p, "reward": reward, "gamma": gamma, "n_iter": max_iter, "epsilon_min": epsilon_min})
        ql_reward = np.mean(forest_policy_reward_cal(n_state, ql.policy, transition_p, reward, gamma, 10000))

        # result_dict[f"{alpha}"]["Number of Iter"] = vi.iter
        # result_dict[f"{alpha}"]["Time"] = vi.time
        result_dict[f"epsilon_min{epsilon_min}"]["Expected Reward"] = str(ql_reward)
        result_dict[f"epsilon_min{epsilon_min}"]["Policy"] = str(ql.policy)

    pd.DataFrame(result_dict).to_excel(f"forest_Experiment_n-state_alpha_epsilon_epsilon_min={n_state}.xlsx")


def forest_policy_reward_cal(n_of_years, policy, P, R, gamma, n_simulation):

    reward_list = []

    # simulate num_iter times of return
    for _ in range(n_simulation):
        reward, cur_state, cur_gamma = (0, 0, 1)
        for step in range(n_of_years):
            cur_action = int(policy[cur_state])
            reward += (R[cur_state][cur_action] * cur_gamma)
            prob = P[cur_action][cur_state]
            cur_state = np.random.choice(list(range(len(prob))), p=prob)
            cur_gamma *= gamma
        reward_list.append(reward)

    return reward_list


def frozen_lake_policy_reward_cal(absorbing_list, policy, P, R, gamma, n_simulation):

    reward_list = []

    # simulate num_iter times of return
    for _ in range(n_simulation):
        # print(_)
        reward, cur_state, new_state, cur_gamma = (0, 0, 0, 1)
        stuck_at_start = 0
        while cur_state not in absorbing_list and stuck_at_start < 100:
            cur_action = int(policy[cur_state])
            prob = P[cur_action][cur_state]
            new_state = np.random.choice(list(range(len(prob))), p=prob)

            reward += (R[cur_action][cur_state][new_state] * cur_gamma)
            cur_gamma *= gamma

            #  if not moving from start for a while
            if cur_state == new_state:
                stuck_at_start += 1
            else:
                stuck_at_start = 0

            cur_state = new_state
        reward_list.append(reward)

    return reward_list


def policy_iteration(**kwargs):
    # pi = ModifiedMDP.mdp.PolicyIteration(**kwargs)
    pi = hiive.mdptoolbox.mdp.PolicyIteration(**kwargs)
    pi.run()
    # print("pi.policy", pi.policy)
    # print("pi.iter", pi.iter)
    # print("pi.time", pi.time)

    return pi


def value_iteration(**kwargs):
    # vi = ModifiedMDP.mdp.ValueIteration(**kwargs)
    vi = hiive.mdptoolbox.mdp.ValueIteration(**kwargs)
    vi.run()
    # print("vi.policy", vi.policy)
    # print("vi.iter", vi.iter)
    # print("vi.time", vi.time)

    return vi


def q_learning(**kwargs):
    ql = hiive.mdptoolbox.mdp.QLearning(**kwargs)
    # ql = ModifiedMDP.mdp.QLearning(**kwargs)
    ql.run()
    print("ql.policy", ql.policy)
    print("ql.V", ql.V)

    return ql


if __name__ == "__main__":
    main()
