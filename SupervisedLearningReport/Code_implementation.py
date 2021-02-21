from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import roc_auc_score, zero_one_loss, accuracy_score
from sklearn.model_selection import learning_curve

from sklearn.datasets import load_breast_cancer

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pickle
import warnings
import datetime
warnings.filterwarnings("ignore")


def process_bank_data(n_data=None):
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
    data_df['age_group'] = pd.cut(data_df['age'], bins=[0, 14, 24, 64, float('inf')], labels=[1, 2, 3, 4], include_lowest=True).astype('uint8')
    data_df['duration_group'] = pd.cut(data_df['duration'], bins=[0, 120, 240, 360, 480, float('inf')], labels=[1, 2, 3, 4, 5], include_lowest=True).astype('uint8')

    data_df.drop(['job', "education", 'marital', 'month', 'day_of_week', 'age', 'duration'], axis=1, inplace=True)

    if n_data is not None:
        data_df = data_df[:n_data]
    # print(data_df.dtypes)
    train, target = data_df[data_df.columns.difference(['y'])], data_df['y']
    return train, target


class MachineLearningModels:
    def __init__(self, data_train, data_target, data_set_name):
        self.data_train = data_train
        self.data_target = data_target
        self.data_set_name = data_set_name

        self.decision_tree_model = DecisionTreeClassifier()
        self.neural_network_model = MLPClassifier(learning_rate='invscaling')
        self.boosting_model = GradientBoostingClassifier()
        self.svm_model = SVC(probability=True)
        self.knn_model = KNeighborsClassifier()
        self.model_name_list = ["decision_tree_model", "neural_network_model", "boosting_model", "svm_model", "knn_model"]

        self.graph_path = os.path.join(os.getcwd(), "Graphs")

    @staticmethod
    def reset_plt(title):
        plt.figure(figsize=(8, 6))
        plt.title(title)

    def plot_dict_data(self, tittle: str, data_dict: dict, xlabel: str, ylabel: str):
        self.reset_plt(tittle)
        for key, value in data_dict.items():
            plt.plot(value, label=key)

        plt.xlabel = xlabel
        plt.ylabel = ylabel
        plt.legend()
        plt.savefig(os.path.join(self.graph_path, f"{tittle}.png"))

    # def training_testing_error_rates(self, training_ratio_list):
    #
    #     training_performance_dict = {}
    #     testing_performance_dict = {}
    #
    #     for model_name in self.model_name_list:
    #         training_performance = []
    #         testing_performance = []
    #
    #         for training_ratio in training_ratio_list:
    #
    #             x_train, x_test, y_train, y_test = train_test_split(self.data_train, self.data_target, train_size=training_ratio)
    #             model = getattr(self, model_name)
    #
    #             model.fit(x_train, y_train)
    #
    #             # training_result = roc_auc_score(y_train, model.predict_proba(x_train)[:, 1])
    #             training_result = 1 - accuracy_score(y_train, model.predict(x_train))
    #             training_performance.append(training_result)
    #
    #             # testing_result = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    #             testing_result = 1 - accuracy_score(y_test, model.predict(x_test))
    #             testing_performance.append(testing_result)
    #
    #             training_performance_dict.update({model_name: training_performance})
    #             testing_performance_dict.update({model_name: testing_performance})
    #
    #     self.plot_dict_data(f"{self.data_set_name}_training_Error_performance", training_performance_dict, xlabel="Percentage of training data", ylabel="Error rate")
    #     self.plot_dict_data(f"{self.data_set_name}_testing_Error_performance", testing_performance_dict, xlabel="Percentage of training data", ylabel="Error rate")

    def learning_curve(self, training_ratio_list):
        if self.data_set_name == "Bank Marketing":
            decision_tree_model = DecisionTreeClassifier(**{'criterion': 'entropy', 'max_depth': 7, 'min_samples_leaf': 40, 'min_samples_split': 40})
            neural_network_model = MLPClassifier(**{'hidden_layer_sizes': (300, 300, 300, 300, 300,), 'learning_rate_init': 0.001, 'tol': 0.006, 'learning_rate': 'invscaling'})
            boosting_model = GradientBoostingClassifier(**{'learning_rate': 0.05, 'loss': 'deviance', 'n_estimators': 100})
            svm_model = SVC(**{'kernel': 'rbf'})
            knn_model = KNeighborsClassifier(**{'n_neighbors': 20})
        else:
            decision_tree_model = DecisionTreeClassifier(**{'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 10, 'min_samples_split': 20})
            neural_network_model = MLPClassifier(**{'hidden_layer_sizes': (300, 300, 300, 300, 300,), 'learning_rate_init': 0.001, 'tol': 0.001, 'learning_rate': 'invscaling'})
            boosting_model = GradientBoostingClassifier(**{'learning_rate': 0.1, 'loss': 'exponential', 'n_estimators': 100})
            svm_model = SVC(**{'kernel': 'rbf'})
            knn_model = KNeighborsClassifier(**{'n_neighbors': 20})
        model_dictionary = dict(zip(self.model_name_list, [decision_tree_model, neural_network_model, boosting_model, svm_model, knn_model]))

        fig, axes = plt.subplots(1, 5, figsize=(20, 6))
        fig.suptitle(f"{self.data_set_name} Dataset", fontsize=20)
        i = 0

        for model_name, model in model_dictionary.items():

            train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, self.data_train, self.data_target, train_sizes=training_ratio_list, cv=5, scoring="accuracy", return_times=True)
            train_scores, test_scores = 1 - train_scores, 1 - test_scores  # make it as error rate
            print(model_name, train_sizes, train_scores, test_scores)

            # plot graph
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
        fig.savefig(os.path.join(self.graph_path, f"{self.data_set_name}_learning_Curve"))

    def final_parameter_selection(self):
        """
        How much performance was due to the problems you chose? How about the values you choose for learning rates, stopping criteria, pruning methods, and so forth
        """
        dt_param = [{
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 3, 4, 5, 6, 7, 8, 9, 10],
            'min_samples_leaf': [1, 10, 20, 40, 60],
            'min_samples_split': [2, 20, 30, 40, 50, 60],
        }]

        neural_network_param = [{
            'hidden_layer_sizes': [(val, val, val, val, val, ) for val in np.arange(2, 110, 50)],
            'tol': [0.00001, 0.001, 0.002, 0.006],
            # 'activation': ['identity', 'logistic', 'tanh', 'relu'],
            # 'solver': ['lbfgs', 'sgd', 'adam'],
            # 'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01, 0.03, 0.04]
        }]

        boosting_param = [{
            'loss': ['deviance', 'exponential'],
            'learning_rate': [0.1, 0.008, 0.01, 0.05],
            'n_estimators': [100, 20, 30, 50, 80, 200],
            'max_depth': [None, 3, 4, 5, 6, 7, 8, 9, 10],
            # 'criterion': ['friedman_mse', 'mse', 'mae'],
            # 'min_samples_leaf': [2, 5, 10, 20, 40]
        }]

        svm_param = [{
            'C': [1.0, 0.6, 0.7, 0.8, 0.9],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            # 'degree': [2, 5, 10, 20, 40],
            # 'gamma': [0.001, 0.0001]
        }]

        knn_param = [{
            'n_neighbors': [5, 10, 20, 100],
            # 'weights': ['uniform', 'distance'],
            # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            # 'leaf_size': [30, 50, 100],
        }]

        model_param_dict = dict(zip(self.model_name_list, [dt_param, neural_network_param, boosting_param, svm_param, knn_param]))

        for model_name, model_param in model_param_dict.items():
            clf = GridSearchCV(getattr(self, model_name), param_grid=model_param,
                                 scoring='roc_auc')

            x_train, x_test, y_train, y_test = train_test_split(self.data_train, self.data_target,
                                                                train_size=0.8)
            clf.fit(x_train, y_train)
            performance_df = pd.DataFrame(clf.cv_results_)
            performance_df.to_excel(os.path.join(self.graph_path, f"final_{self.data_set_name}_{model_name}.xlsx"))

            print(model_name, clf.best_estimator_, clf.best_score_)

    def parameter_learning_curve(self):
        """
        How much performance was due to the problems you chose? How about the values you choose for learning rates, stopping criteria, pruning methods, and so forth
        """
        dt_param = {
            'max_depth': np.arange(2, 100, 1),
            'min_samples_split': np.arange(2, 100, 1),
            'min_samples_leaf': np.arange(2, 100, 1),
            'min_weight_fraction_leaf': np.arange(0.0, 0.9, 0.01),
            'min_impurity_decrease': np.arange(0.0, 0.9, 0.01),
        }

        neural_network_param = {
            'hidden_layer_sizes': [(val, val, val, val, val, ) for val in np.arange(2, 400, 50)],
            'alpha': np.arange(0, 0.01, 0.0004),
            'learning_rate_init': np.arange(0, 0.1, 0.004),
            'max_iter': np.arange(100, 500, 20),
            'tol': np.arange(0.0001, 0.01, 0.0004),
        }

        boosting_param = {
            'learning_rate': np.arange(0.0001, 0.01, 0.0001),
            'n_estimators': np.arange(10, 1000, 10),
            'min_samples_split': np.arange(2, 100, 1),
            'min_samples_leaf': np.arange(1, 100, 1),
            'max_depth': np.arange(1, 100, 1),
            'min_impurity_decrease': np.arange(0.0, 0.9, 0.01),
        }

        svm_param = {
            'C': np.arange(0.0, 1.0, 0.01),
            'tol': np.arange(0.0001, 0.01, 0.0001),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        }

        knn_param = {
            'n_neighbors': np.arange(2, 100, 1),
            'leaf_size': np.arange(2, 100, 1)
        }

        model_param_dict = dict(zip(self.model_name_list, [dt_param, neural_network_param, boosting_param, svm_param, knn_param]))

        for model_name, model_param in model_param_dict.items():
            for key, value in model_param.items():

                # clf = GridSearchCV(getattr(self, model_name), param_grid={key: value},
                #                      scoring='roc_auc')
                #
                # x_train, x_test, y_train, y_test = train_test_split(self.data_train, self.data_target,
                #                                                     train_size=0.8)
                # clf.fit(x_train, y_train)
                # performance_df = pd.DataFrame(clf.cv_results_)
                #
                # print(datetime.datetime.now(),model_name, key, clf.best_estimator_, clf.best_score_)
                #
                # performance_df.to_excel(os.path.join(self.graph_path, f"{self.data_set_name}_{model_name}_{key}.xlsx"))
                # filehandler = open(os.path.join(self.graph_path, f"{self.data_set_name}_{model_name}_{key}"), "wb")
                # pickle.dump(performance_df, filehandler)
                # filehandler.close()

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

    def parameter_learning_curve_together(self, data_train_2, data_target_2, data_set_name_2):
        """
        How much performance was due to the problems you chose? How about the values you choose for learning rates, stopping criteria, pruning methods, and so forth
        """
        dt_param = {
            'max_depth': np.arange(2, 100, 1),
            'min_samples_split': np.arange(2, 350, 4),
            'min_samples_leaf': np.arange(2, 100, 1),
            'min_weight_fraction_leaf': np.arange(0.0, 0.9, 0.01),
            'min_impurity_decrease': np.arange(0.0, 0.9, 0.01),
        }

        neural_network_param = {
            'hidden_layer_sizes': [(val, val, val, val, val, ) for val in np.arange(2, 400, 50)],
            'alpha': np.arange(0, 0.01, 0.0004),
            'learning_rate_init': np.arange(0, 0.1, 0.004),
            'max_iter': np.arange(100, 500, 15),
            'tol': np.arange(0.0001, 0.01, 0.0004),
        }

        boosting_param = {
            'learning_rate': np.arange(0.001, 0.5, 0.01),
            'n_estimators': np.arange(10, 1000, 10),
            'min_samples_split': np.arange(2, 100, 1),
            'min_samples_leaf': np.arange(1, 100, 1),
            'max_depth': np.arange(1, 100, 1),
            'min_impurity_decrease': np.arange(0.0, 0.9, 0.01),
        }

        svm_param = {
            'C': np.arange(0.0, 1.0, 0.01),
            'tol': np.arange(0.0001, 0.01, 0.0001),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        }

        knn_param = {
            'n_neighbors': np.arange(2, 350, 4),
            'leaf_size': np.arange(2, 100, 1)
        }

        # model_param_dict = dict(zip(self.model_name_list, [dt_param, neural_network_param, boosting_param, svm_param, knn_param]))
        model_param_dict = dict(zip(["decision_tree_model"], [dt_param]))

        for model_name, model_param in model_param_dict.items():
            for key, value in model_param.items():

                def get_info(data_train, data_target, graph_path, data_set_name):
                    clf = GridSearchCV(getattr(self, model_name), param_grid={key: value},
                                       scoring='roc_auc')

                    x_train, x_test, y_train, y_test = train_test_split(data_train, data_target,
                                                                        train_size=0.8)
                    clf.fit(x_train, y_train)
                    performance_df = pd.DataFrame(clf.cv_results_)

                    print(datetime.datetime.now(),model_name, key, clf.best_estimator_, clf.best_score_)

                    performance_df.to_excel(os.path.join(graph_path, f"{data_set_name}_{model_name}_{key}.xlsx"))
                    filehandler = open(os.path.join(graph_path, f"{data_set_name}_{model_name}_{key}"), "wb")
                    pickle.dump(performance_df, filehandler)
                    filehandler.close()

                    # filehandler = open(os.path.join(graph_path, f"{data_set_name}_{model_name}_{key}"), "rb")
                    # performance_df = pickle.load(filehandler)
                    # filehandler.close()

                    if key == "hidden_layer_sizes":
                        x_value = np.array([i[0] for i in performance_df[f"param_{key}"].values], dtype=float)
                    else:
                        try:
                            x_value = np.array(performance_df[f"param_{key}"], dtype=float)
                        except:
                            x_value = performance_df[f"param_{key}"].values

                    score_mean, score_std = np.array(performance_df.mean_test_score, dtype=float), np.array(performance_df.std_test_score, dtype=float)
                    fit_times_mean, fit_times_std = performance_df.mean_fit_time, performance_df.std_fit_time

                    return x_value, score_mean, score_std, fit_times_mean, fit_times_std

                x_value_1, score_mean_1, score_std_1, fit_times_mean_1, fit_times_std_1 = get_info(self.data_train, self.data_target, self.graph_path, self.data_set_name)
                x_value_2, score_mean_2, score_std_2, fit_times_mean_2, fit_times_std_2 = get_info(data_train_2, data_target_2, self.graph_path, data_set_name_2)

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                fig.suptitle(f"{model_name} parameter : {key}", fontsize=20)
                axes[0].grid()
                axes[0].set_title(f"roc_auc_score curve")
                # axes[0].set_ylim(0.5, 1)
                axes[0].set_xlabel(key)
                axes[0].set_ylabel("roc_auc_score")
                axes[0].fill_between(x_value_1, score_mean_1 - score_std_1,
                                     score_mean_1 + score_std_1, alpha=0.1, color="g")
                axes[0].fill_between(x_value_2, score_mean_2 - score_std_2,
                                     score_mean_2 + score_std_2, alpha=0.1, color="orange")
                axes[0].plot(x_value_1, score_mean_1, 'o-', color="g", label=f"{self.data_set_name} dataset")
                axes[0].plot(x_value_2, score_mean_2, 'o-', color="orange", label=f"{data_set_name_2} dataset")
                axes[0].legend(loc="best")

                axes[1].grid()
                axes[1].plot(x_value_1, fit_times_mean_1, 'o-', color="g", label=f"{self.data_set_name} dataset")
                axes[1].plot(x_value_2, fit_times_mean_2, 'o-', color="orange", label=f"{data_set_name_2} dataset")
                axes[1].fill_between(x_value_1, fit_times_mean_1 - fit_times_std_1,
                                     fit_times_mean_1 + fit_times_std_1, alpha=0.1, color="g")
                axes[1].fill_between(x_value_2, fit_times_mean_2 - fit_times_std_2,
                                     fit_times_mean_2 + fit_times_std_2, alpha=0.1, color="orange")
                axes[1].set_xlabel(key)
                axes[1].set_ylabel("fit_times in seconds")
                axes[1].set_title("Scalability of the model")
                axes[1].legend(loc="best")

                fig.tight_layout()
                fig.savefig(os.path.join(self.graph_path, f"Both Dataset_{model_name}_Param_{key}_learning_Curve"))

    ## not useful anymore
    # def performance_comparison(self, data_train, data_target, data_set_name):
    #     if self.data_set_name == "Bank Marketing":
    #         decision_tree_model = DecisionTreeClassifier(**{'criterion': 'entropy', 'max_depth': 7, 'min_samples_leaf': 40, 'min_samples_split': 40})
    #         neural_network_model = MLPClassifier(**{'hidden_layer_sizes': (300, 300, 300, 300, 300,), 'learning_rate_init': 0.001, 'tol': 0.006, 'learning_rate': 'invscaling'})
    #         boosting_model = GradientBoostingClassifier(**{'learning_rate': 0.05, 'loss': 'deviance', 'n_estimators': 100})
    #         svm_model = SVC(**{'kernel': 'rbf'})
    #         knn_model = KNeighborsClassifier(**{'n_neighbors': 20})
    #     else:
    #         decision_tree_model = DecisionTreeClassifier(**{'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 10, 'min_samples_split': 20})
    #         neural_network_model = MLPClassifier(**{'hidden_layer_sizes': (300, 300, 300, 300, 300,), 'learning_rate_init': 0.001, 'tol': 0.001, 'learning_rate': 'invscaling'})
    #         boosting_model = GradientBoostingClassifier(**{'learning_rate': 0.1, 'loss': 'exponential', 'n_estimators': 100})
    #         svm_model = SVC(**{'kernel': 'rbf'})
    #         knn_model = KNeighborsClassifier(**{'n_neighbors': 20})
    #
    #     trained_model_dictionary = dict(zip(self.model_name_list, [decision_tree_model, neural_network_model, boosting_model, svm_model, knn_model]))
    #     base_model_dictionary = dict(zip(self.model_name_list, [DecisionTreeClassifier(), MLPClassifier(), GradientBoostingClassifier(), SVC(), KNeighborsClassifier()]))
    #
    #     for model_name, base_model in base_model_dictionary.items():
    #
    #         trained_model = trained_model_dictionary[model_name]
    #         x_train, x_test, y_train, y_test = train_test_split(data_train, data_target, train_size=0.8)
    #
    #         base_model.fit(x_train, y_train)
    #         y_predict = base_model.predict(x_test)
    #         print(data_set_name, model_name, "base_model",roc_auc_score(y_test, y_predict))
    #
    #         trained_model.fit(x_train, y_train)
    #         y_predict = trained_model.predict(x_test)
    #         print(data_set_name, model_name, "trained_model",roc_auc_score(y_test, y_predict), "\n")


if __name__ == "__main__":
    bank_data_train, bank_data_target = process_bank_data()
    breast_cancer_train, breast_cancer_target = load_breast_cancer().data, load_breast_cancer().target

    #
    model_experiment = MachineLearningModels(breast_cancer_train, breast_cancer_target, "Breast Cancer")
    model_experiment.learning_curve(np.arange(0.1, 0.9, 0.01))

    model_experiment = MachineLearningModels(bank_data_train, bank_data_target, "Bank Marketing")
    model_experiment.learning_curve(np.arange(0.1, 0.9, 0.05))
    model_experiment.parameter_learning_curve_together(breast_cancer_train, breast_cancer_target, "Breast Cancer")


    # final model selection
    model_experiment = MachineLearningModels(bank_data_train, bank_data_target, "Bank Marketing")
    model_experiment.final_parameter_selection()
    model_experiment = MachineLearningModels(breast_cancer_train, breast_cancer_target, "Breast Cancer")
    model_experiment.final_parameter_selection()

