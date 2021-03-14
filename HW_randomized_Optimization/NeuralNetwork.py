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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.modules['sklearn.externals.six'] = six
import mlrose
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve


class NeuralNetwork:
    def __init__(self, n_data=None):
        self.data_train, self.data_target = self.process_bank_data(n_data=n_data)

    def run_experiment(self):
        # from my setting
        hidden_nodes = [300, 300, 300, 300, 300]
        learning_rate = 0.005
        tol = 0.006

        # default from Sklearn and
        activation = 'relu'
        max_iters = 200

        # # from mlrose
        # bias = True
        # is_classifier = True
        clip_max = 5

        random_hill_climb = mlrose.NeuralNetwork(hidden_nodes=[300, 300, 300, 300, 300],early_stopping=True, clip_max=clip_max, learning_rate=learning_rate, algorithm='random_hill_climb', random_state=3)
        simulated_annealing = mlrose.NeuralNetwork(hidden_nodes=[300, 300, 300, 300, 300],early_stopping=True, clip_max=clip_max, learning_rate=learning_rate, algorithm='simulated_annealing', random_state=3, schedule=mlrose.GeomDecay())
        genetic_alg = mlrose.NeuralNetwork(hidden_nodes=[300, 300, 300, 300, 300],early_stopping=True, clip_max=clip_max, learning_rate=learning_rate, algorithm='genetic_alg', random_state=3, pop_size=100, mutation_prob=0.15)
        backprop = MLPClassifier(**{'hidden_layer_sizes': (300, 300, 300, 300, 300,), 'learning_rate_init': 0.001, 'tol': 0.006, 'learning_rate': 'invscaling'})

        result_list = []
        model_dictionary = dict(zip(["random_hill_climb", "simulated_annealing", "genetic_alg", "backprop"], [random_hill_climb, simulated_annealing, genetic_alg, backprop]))
        for model_name, model in model_dictionary.items():
            x_train, x_test, y_train, y_test = train_test_split(self.data_train, self.data_target, train_size=0.8)
            start = time.time()
            model.fit(x_train, y_train)
            end = time.time()

            model.fit(x_train, y_train)
            y_train_pred = model.predict(x_train)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            y_test_pred = model.predict(x_test)
            y_test_accuracy = accuracy_score(y_test, y_test_pred)

            f_one_score = f1_score(y_test, y_test_pred)

            if model_name != "backprop":
                roc_score = roc_auc_score(y_test, model.predicted_probs)
            else:
                roc_score = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

            result_dict = {"model": model_name, "training_accuracy": y_train_accuracy, "testing_accuracy": y_test_accuracy, "f1_score": f_one_score, "roc_score": roc_score, "training_time": end-start}
            result_list.append(result_dict)

            pd.DataFrame(result_list).to_excel("Final_model_Performance.xlsx")

    def learning_curve(self, training_ratio_list):
        random_hill_climb = mlrose.NeuralNetwork(hidden_nodes=[300, 300, 300, 300, 300], activation='relu',algorithm='random_hill_climb', max_attempts=100, random_state=3)
        simulated_annealing = mlrose.NeuralNetwork(hidden_nodes=[300, 300, 300, 300, 300], activation='relu',algorithm='simulated_annealing', max_attempts=100, random_state=3)
        genetic_alg = mlrose.NeuralNetwork(hidden_nodes=[300, 300, 300, 300, 300], activation='relu',algorithm='genetic_alg', max_attempts=100, random_state=3)
        backprop = MLPClassifier(**{'hidden_layer_sizes': (300, 300, 300, 300, 300,), 'learning_rate_init': 0.001, 'tol': 0.006, 'learning_rate': 'invscaling'})


        model_dictionary = dict(zip(["random_hill_climb", "simulated_annealing", "genetic_alg", "backprop"], [random_hill_climb, simulated_annealing, genetic_alg, backprop]))

        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        fig.suptitle(f"Learning Curve Plot", fontsize=20)
        i = 0

        for model_name, model in model_dictionary.items():

            if model_name == "backprop":
                train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, self.data_train, self.data_target, train_sizes=training_ratio_list, cv=5, scoring="accuracy", return_times=True)
                train_scores, test_scores = 1 - train_scores, 1 - test_scores  # make it as error rate
                print(model_name, train_sizes, train_scores, test_scores)
                # plot graph

            else:
                train_sizes, train_scores, test_scores, fit_times = list(), list(), list(), list()
                for train_ratio in training_ratio_list:
                    train_sizes_temp, train_scores_temp, test_scores_temp, fit_times_temp = list(), list(), list(), list()
                    for cv in range(5):

                        x_train, x_test, y_train, y_test = train_test_split(self.data_train, self.data_target, train_size=train_ratio)
                        start = time.time()
                        model.fit(x_train, y_train)
                        end = time.time()

                        y_train_pred = model.predict(x_train)
                        y_train_accuracy = accuracy_score(y_train, y_train_pred)
                        y_test_pred = model.predict(x_test)
                        y_test_accuracy = accuracy_score(y_test, y_test_pred)

                        train_sizes_temp.append(len(y_train))
                        train_scores_temp.append(1 - y_train_accuracy)
                        test_scores_temp.append(1 - y_test_accuracy)
                        fit_times_temp.append(end - start)

                    train_sizes.append(train_sizes_temp)
                    train_scores.append(train_scores_temp)
                    test_scores.append(test_scores_temp)
                    fit_times.append(fit_times_temp)

            train_sizes = np.mean(train_sizes, axis=1)
            train_scores_mean, train_scores_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
            test_scores_mean, test_scores_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)


            axes[i].grid()
            axes[i].set_title(f"{model_name}")
            axes[i].set_ylim(0, 0.2)
            axes[i].set_xlabel("Training examples")
            axes[i].set_ylabel("Error rate")
            axes[i].fill_between(train_sizes, train_scores_mean - train_scores_std,
                                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
            axes[i].fill_between(train_sizes, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
            axes[i].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training error rate")
            axes[i].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Testing error rate")
            axes[i].legend(loc="best")

            i += 1
        fig.savefig(os.path.join("Graph", "LearningCurve"))

    def parameter_learning_curve(self):
        """
        How much performance was due to the problems you chose? How about the values you choose for learning rates, stopping criteria, pruning methods, and so forth
        """
        genetic_alg_param = {
            'pop_size': np.arange(10, 100, 15),
            'mutation_prob': np.arange(0.1, 0.7, 0.1),
        }

        simulated_annealing_param = {
            'decay': [mlrose.GeomDecay(), mlrose.ArithDecay(), mlrose.ExpDecay()],
        }

        mimic_param = {
            'pop_size': np.arange(10, 100, 15),
            'keep_pct': np.arange(0.1, 0.7, 0.1),
        }

        model_param_dict = dict(zip([], [genetic_alg_param, simulated_annealing_param, mimic_param]))

        for model_name, model_param in model_param_dict.items():
            for key, value in model_param.items():

                filehandler = open(os.path.join(self.graph_path, f"{self.data_set_name}_{model_name}_{key}"), "wb")
                performance_df = pickle.load(filehandler)
                filehandler.close()

                if key == "hidden_layer_sizes":
                    x_value = np.array([i[0] for i in performance_df[f"param_{key}"].values], dtype=float)
                else:
                    try:
                        x_value = np.array(performance_df[f"param_{key}"], dtype=float)
                    except:
                        x_value = performance_df[f"param_{key}"].values

                score_mean, score_std = np.array(performance_df.mean_test_score, dtype=float), np.array(performance_df.std_test_score, dtype=float)
                fit_times_mean, fit_times_std = performance_df.mean_fit_time, performance_df.std_fit_time

                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                fig.suptitle(f"{self.data_set_name} Dataset", fontsize=20)
                axes[0].grid()
                axes[0].set_title(f"roc_auc_score for {model_name} model parameter : {key}")
                axes[0].set_ylim(0.5, 1)
                axes[0].set_xlabel(key)
                axes[0].set_ylabel("roc_auc_score")
                axes[0].fill_between(x_value, score_mean - score_std,
                                     score_mean + score_std, alpha=0.1, color="g")
                axes[0].plot(x_value, score_mean, 'o-', color="g", label="Testing error rate")
                axes[0].legend(loc="best")

                axes[1].grid()
                axes[1].plot(x_value, fit_times_mean, 'o-')
                axes[1].fill_between(x_value, fit_times_mean - fit_times_std,
                                     fit_times_mean + fit_times_std, alpha=0.1)
                axes[1].set_xlabel(key)
                axes[1].set_ylabel("fit_times in seconds")
                axes[1].set_title("Scalability of the model")

                fig.savefig(os.path.join(self.graph_path, f"{self.data_set_name}_{model_name}_Param_{key}_learning_Curve"))

    def process_bank_data(self, n_data=None):
        data_df = pd.read_csv("bank-additional.csv", sep=";")

        # convert them to binary
        data_df['y'] = data_df['y'].map({'no': 0, 'yes': 1}).astype('uint8')
        data_df["default"] = data_df["default"].map({'no': 1, 'unknown': 0, 'yes': 0}).astype('uint8')
        data_df["housing"] = data_df["housing"].map({'yes': 1, 'unknown': 0, 'no': 0}).astype('uint8')
        data_df["loan"] = data_df["loan"].map({'yes': 1, 'unknown': 0, 'no': 0}).astype('uint8')
        data_df["contact"] = data_df["contact"].map({'cellular': 1, 'telephone': 0}).astype('uint8')
        data_df["pdays"] = data_df["pdays"].replace(999, 0).astype('uint8')
        data_df["previous"] = data_df["previous"].apply(lambda x: 1 if x > 0 else 0).astype('uint8')
        data_df["poutcome"] = data_df["poutcome"].map({'nonexistent': 0, 'failure': 0, 'success': 1}).astype('uint8')

        # normalized to start from 0
        data_df['cons.price.idx'] = (data_df['cons.price.idx'] * 10).astype('uint8')
        data_df['cons.price.idx'] = data_df['cons.price.idx'] - data_df['cons.price.idx'].min()

        data_df['cons.conf.idx'] = data_df['cons.conf.idx'] * -1
        data_df['cons.conf.idx'] = data_df['cons.conf.idx'] - data_df['cons.conf.idx'].min()

        # log transformation
        data_df['nr.employed'] = np.log2(data_df['nr.employed']).astype('uint8')

        data_df["euribor3m"] = data_df["euribor3m"].astype('uint8')
        data_df["campaign"] = data_df["campaign"].astype('uint8')
        data_df["pdays"] = data_df["pdays"].astype('uint8')

        data_df = pd.concat([data_df, pd.get_dummies(data_df["job"], prefix="job")], axis=1)
        data_df = pd.concat([data_df, pd.get_dummies(data_df["education"], prefix="education")], axis=1)
        data_df = pd.concat([data_df, pd.get_dummies(data_df["marital"], prefix="marital")], axis=1)
        data_df = pd.concat([data_df, pd.get_dummies(data_df["month"], prefix="month")], axis=1)
        data_df = pd.concat([data_df, pd.get_dummies(data_df["day_of_week"], prefix="day_of_week")], axis=1)
        data_df['age_group'] = pd.cut(data_df['age'], bins=[0, 14, 24, 64, float('inf')], labels=[1, 2, 3, 4],
                                      include_lowest=True).astype('uint8')
        data_df['duration_group'] = pd.cut(data_df['duration'], bins=[0, 120, 240, 360, 480, float('inf')],
                                           labels=[1, 2, 3, 4, 5], include_lowest=True).astype('uint8')

        data_df.drop(['job', "education", 'marital', 'month', 'day_of_week', 'age', 'duration'], axis=1, inplace=True)

        if n_data is not None:
            data_df = data_df[:n_data]
        # print(data_df.dtypes)
        train, target = data_df[data_df.columns.difference(['y'])], data_df['y']
        return train, target


if __name__ == "__main__":
    nn = NeuralNetwork(n_data=1000)
    nn.learning_curve(training_ratio_list=np.arange(0.1, 0.8, 0.2))
    # nn.run_experiment()