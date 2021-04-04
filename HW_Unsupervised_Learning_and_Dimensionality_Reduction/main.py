from Process_Data import bank_data, digits_data, bank_data_original

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as EM
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection as RandomProjection
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LassoCV
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import os
import pickle
import warnings
from scipy.stats import mode
import numpy as np
from collections import defaultdict
import seaborn as sns
import copy
import time
from collections import Counter
# sns.set(style="ticks")


class UnsupervisedLearning:
    def __init__(self):
        self.data_dict = self._get_data()
        self.graph_path = "Graph"
        self.random_state = 0
        self.max_k = 20

        self.pair_plot()
        # self.elbow_method(self.data_dict, "Original", "KMeans", self.max_k)
        self.cluster(self.data_dict, "Original", "KMeans")
        self.cluster(self.data_dict, "Original", "EM")
        self.silhouette_score(self.data_dict, "Original", "KMeans", self.max_k) # Yes
        self.silhouette_score(self.data_dict, "Original", "EM", self.max_k) # Yes

        self.pca_distribution_of_eigenvalues(self.data_dict, "Original") # Yes
        self.ica_kurtotic(self.data_dict, "Original") # Yes
        self.reconstruction_error(self.data_dict, "Original", algo_name="PCA", n_times=10) # Yes
        self.reconstruction_error(self.data_dict, "Original", algo_name="RandomProjection", n_times=10) # Yes
        self.feature_importance_lasso_cv(self.data_dict, "Original")

        self.rerun_neural_network()
        self.rerun_neural_network_clustering()
        self.roc_score_4_5()
        # self.elbow_method(self.data_dict, "Original", "EM", self.max_k)
        # self.dimension_reduction()

    @staticmethod
    def _get_data():
        bank_train, bank_target = bank_data()
        digit_train, digit_target = digits_data()

        return {"Bank_data": [bank_train, bank_target, 2], "Digit_Data": [digit_train, digit_target, 10]}

    def pair_plot(self):

        df = bank_data_original()
        df = df[['duration','cons.conf.idx','nr.employed','contact', 'y']]
        sns_plot = sns.pairplot(df, dropna=True, hue='y')
        sns_plot.savefig(os.path.join(self.graph_path, "Pair plot of Bank Data"))

    """
    1.	Run the clustering algorithms on the datasets and describe what you see.
    explore different K and see which K is best to choice 
    """
    def elbow_method(self, data_dict, file_name, algo_name, k_max):
        for name, val in data_dict.items():
            train, target, _ = val[0], val[1], val[2]
            sse = []
            for k in range(1, k_max):
                cluster_algo = KMeans(k, random_state=self.random_state) if algo_name == "KMeans" else EM(k, random_state=self.random_state)
                cluster_algo.fit_predict(train)
                sse.append(cluster_algo.inertia_)

            plt.figure(figsize=(8, 6))
            plt.plot(list(range(1, k_max)), sse, '-o')
            plt.xlabel(r'Number of clusters *k*')
            plt.ylabel('Sum of squared distance')
            full_file_name = f"SSE for {algo_name} in {file_name} {name} dataset with different clusters K"
            plt.title(full_file_name)
            plt.savefig(os.path.join(self.graph_path, full_file_name))
            plt.close()

    def silhouette_score(self, data_dict, file_name, algo_name, k_max):
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        full_file_name = f"silhouette_score for {algo_name} in {file_name} dataset with different clusters K"
        fig.suptitle(full_file_name, fontsize=20)
        max_score_index_dict = {}
        i = 0

        for name, val in data_dict.items():
            train, target, _ = val[0], val[1], val[2]
            sil = []
            for k in range(2, k_max):
                cluster_algo = KMeans(k, random_state=self.random_state) if algo_name == "KMeans" else EM(k, random_state=self.random_state)
                labels = cluster_algo.fit_predict(train)
                sil.append(silhouette_score(train, labels, metric='euclidean'))

            max_score_index = np.argmax(sil)+2
            max_score_index_dict[name] = max_score_index
            axes[i].axvline(x=max_score_index, label=f'max silhouette score, n_cluster = {max_score_index}', color='orange', linestyle='--')
            axes[i].set_title(f"{name} dataset")
            # axes[i].set_ylim(0, 1)
            axes[i].set_xlabel("Number of clusters *k*")
            axes[i].set_ylabel("silhouette_score")
            axes[i].plot(list(range(2, k_max)), sil, '-o')
            axes[i].legend(loc="upper left")
            axes[i].grid()
            i += 1

        plt.savefig(os.path.join(self.graph_path, full_file_name))
        plt.close()

        # plot best K based on silhouette_score
        # if algo_name == "KMeans":
        for name, max_score_index in max_score_index_dict.items():
            train, target, _ = data_dict[name]
            # cluster_algo = KMeans(max_score_index, random_state=self.random_state)
            cluster_algo = KMeans(max_score_index, random_state=self.random_state) if algo_name == "KMeans" else EM(max_score_index,random_state=self.random_state)
            labels = cluster_algo.fit_predict(train)

            tsne = TSNE(**{"n_components": 2, "random_state": self.random_state})
            data_projected = tsne.fit_transform(train)
            self.plot_scatter(data_projected, labels, f"{algo_name} on {file_name} visualization scatter plot for {name}, n_cluster = {max_score_index}")

    def pca_distribution_of_eigenvalues(self, data_dict, file_name, percentage=0.95):
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        fig.suptitle(f"PCA eigenvalues value explained variance distribution", fontsize=20)
        i = 0

        new_data_dict = {}

        for name, val in data_dict.items():

            train, target, _ = val[0], val[1], val[2]
            sse = []
            n_list = list(range(1, train.shape[1]))
            print(n_list)
            smallest = True
            for n in n_list:
                my_model = PCA(n_components=n, random_state=self.random_state)
                train_projected = my_model.fit_transform(train)
                variance = sum(my_model.explained_variance_ratio_)
                if variance>percentage and smallest:
                    axes[i].axvline(x=n, label=f'{percentage*100}% explained variance, n_component = {n}')
                    smallest = False
                    new_data_dict.update({name: [train_projected, target, _]})
                sse.append(variance)

            axes[i].set_title(f"{name} dataset")
            axes[i].set_ylim(0, 1)
            axes[i].set_xlabel("N of components")
            axes[i].set_ylabel("Percentage of variance explained")
            axes[i].plot(n_list, sse, 'o-', color="r", label=f"Percentage of variance explained on {name} dataset")
            axes[i].legend(loc="upper left")
            axes[i].grid()
            i += 1

        plt.savefig(os.path.join(self.graph_path, f"PCA_distribution_of_eigenvalues on {file_name} dataset"))
        plt.close()

        self.silhouette_score(new_data_dict, "PCA Reduced", "KMeans", self.max_k)
        self.silhouette_score(new_data_dict, "PCA Reduced", "EM", self.max_k)

    def ica_kurtotic(self, data_dict, file_name):
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        fig.suptitle(f"ICA average kurtotic distribution", fontsize=20)
        i = 0
        new_data_dict = {}

        for name, val in data_dict.items():

            train, target, _ = val[0], val[1], val[2]
            sse = []
            n_list = list(range(1, train.shape[1]))
            print(n_list)

            for n in n_list:
                my_model = ICA(n_components=n, random_state=self.random_state, max_iter=1000, tol=0.001)
                X_transformed = my_model.fit_transform(train)
                kurtotic_list = [kurtosis(i) for i in X_transformed.T]
                sse.append(np.mean(kurtotic_list))

            print(f"kurtosis for {name} dataset {sse}")
            max_score_index = np.argmax(sse)+1
            my_model = ICA(n_components=max_score_index, random_state=self.random_state, max_iter=1000, tol=0.001)
            X_transformed = my_model.fit_transform(train)
            new_data_dict.update({name: [X_transformed, target, _]})

            axes[i].axvline(x=max_score_index, label=f'max kurtotic, n_component = {max_score_index}', color='orange', linestyle='--')
            axes[i].set_title(f"{name} dataset")
            axes[i].set_xlabel("N of components")
            axes[i].set_ylabel("average kurtosis")
            axes[i].plot(n_list, sse, 'o-', color="r", label="average kurtosis")
            axes[i].legend(loc="upper left")
            axes[i].grid()
            i += 1

        plt.savefig(os.path.join(self.graph_path, f"ICA average kurtotic on {file_name} dataset"))
        plt.close()

        self.silhouette_score(new_data_dict, "ICA Reduced", "KMeans", self.max_k)
        self.silhouette_score(new_data_dict, "ICA Reduced", "EM", self.max_k)

    def reconstruction_error(self, data_dict, file_name, algo_name="PCA", n_times=10):
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        fig.suptitle(f"Reconstruction Error for {algo_name}", fontsize=20)
        i = 0
        new_data_dict = {}

        for name, val in data_dict.items():

            train, target, _ = val[0], val[1], val[2]
            n_list = list(range(1, train.shape[1]))
            print(n_list)
            mse_whole = []
            for experiment in range(n_times):
                mse = []

                for n in n_list:
                    my_model = PCA(n_components=n) if algo_name == "PCA" else RandomProjection(n_components=n)
                    X_transformed = my_model.fit_transform(train)
                    revert_x = np.dot(X_transformed, np.linalg.pinv(my_model.components_.T))
                    mean_error = mean_squared_error(train, revert_x)
                    mse.append(mean_error)

                    if 0.04 < mean_error < 0.06:
                        new_data_dict.update({name: [X_transformed, target, _]})

                mse_whole.extend(mse)
                axes[i].plot(n_list, mse, label=f"{experiment}th experiment")

            x = np.array(n_list*n_times)
            print(x, mse_whole)
            m, b = np.polyfit(n_list*n_times, mse_whole, 1)
            axes[i].plot(x, m * x + b, color='black', linestyle='--', label=f"trend line")
            axes[i].set_title(f"{name} dataset")
            axes[i].set_xlabel("N of components")
            axes[i].set_ylabel("Reconstruction Error")
            axes[i].legend(loc="upper right")
            axes[i].grid()
            i += 1

        plt.savefig(os.path.join(self.graph_path, f"Reconstruction Error for {algo_name} on {file_name} dataset"))
        plt.close()

        self.silhouette_score(new_data_dict, "Randomized Projection Reduced", "KMeans", self.max_k)
        self.silhouette_score(new_data_dict, "Randomized Projection Reduced", "EM", self.max_k)

    def feature_importance_lasso_cv(self, data_dict, file_name):
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        fig.suptitle(f"Feature importance distribution using LassoCV", fontsize=20)
        i = 0
        new_data_dict = {}

        for name, val in data_dict.items():

            train, target, _ = val[0], val[1], val[2]

            lasso = LassoCV().fit(train, target)
            importance = np.abs(lasso.coef_)
            feature_names = np.array(list(train))

            if name == "Bank_data":
                n_feature = (importance > 0.01).sum()
                sfs_forward = SequentialFeatureSelector(lasso, n_features_to_select=n_feature, direction='forward').fit(
                    train, target)
                feature_selected = feature_names[sfs_forward.get_support()]
                print("Features selected by forward sequential selection: "
                      f"{feature_selected}")
                new_data_dict.update({name: [train[feature_selected], target, _]})


                # sorted_importance = sorted(zip(importance, feature_names), key=lambda x: x[0], reverse=True)[:n_feature]
                # importance = [i[0] for i in sorted_importance]
                # feature_names = [i[1] for i in sorted_importance]
                # axes[i].set_xticklabels(feature_names, rotation=45)
            else:
                n_feature = (importance > 0.1).sum()
                new_data_dict.update({name: [train, target, _]})

            print("digit data max_n", n_feature)
            axes[i].set_title(f"{name} dataset")
            axes[i].set_xlabel("Feature")
            axes[i].set_ylabel("Feature Importance")
            axes[i].bar(height=importance, x=list(range(train.shape[1])))
            axes[i].legend(loc="upper left")
            axes[i].grid()
            i += 1

        plt.savefig(os.path.join(self.graph_path, f"Feature importance distribution using LassoCV on {file_name} dataset"))
        plt.close()

        self.silhouette_score(new_data_dict, "Forward Selection Reduced", "KMeans", self.max_k)
        self.silhouette_score(new_data_dict, "Forward Selection Reduced", "EM", self.max_k)


    def cluster(self, data_dict, file_name, algo_name, performance_dict=None):

        for name, val in data_dict.items():
            train, target, k = val[0], val[1], val[2]

            # x_train, x_test, y_train, y_test = train_test_split(train, target, train_size=0.8)
            # k = dr_dict[name][0] if algo_name == "KMeans" else dr_dict[name][1]
            cluster_algo = KMeans(k, random_state=self.random_state) if algo_name == "KMeans" else EM(k, random_state=self.random_state)
            clusters = cluster_algo.fit_predict(train)

            labels = np.zeros_like(clusters)
            add_set = set()
            for i in range(k):
                mask = (clusters == i)

                temp = 1
                counts = Counter(target[mask])
                predict = counts.most_common(temp)[-1][0]

                while predict in add_set:
                    temp += 1
                    predict = counts.most_common(temp)[-1][0]
                add_set.add(predict)

                # print(mode(target[mask])[0], name)
                # print(predict, name)
                labels[mask] = predict

            score = accuracy_score(target, labels)
            print(f"{algo_name} accuracy_score for {file_name}  {score} for {name}")

            # there can be some case performance_dict was not passed in
            if performance_dict is not None:
                performance_dict[name]["accuracy_score"].append(score)

            self.plot_heat_map(target, labels, f"{algo_name} {file_name} accuracy map for {name}")

        return performance_dict

    def dimension_reduction(self):

        dr_dict = {"PCA": PCA, "ICA": ICA, "Random_Projection": RandomProjection, "TSNE": TSNE}

        # iterate 5 dimension reduction algorithm, and project the data to 2 dimension space for each algorithm
        # then call the classification algorithm to see the difference
        for algo_name, algo in dr_dict.items():
            data_dict = dict()
            k_means_performance_dict = defaultdict(lambda: defaultdict(list))
            em_performance_dict = defaultdict(lambda: defaultdict(list))

            # TSNE only support up to 4 dimension
            n_range = np.arange(1, 4) if algo_name == "TSNE" else np.arange(1, 50)
            # n_range = np.arange(1, 4)

            for n_components in n_range:

                for name, val in self.data_dict.items():
                    train, target, k = val[0], val[1], val[2]

                    algo_trained = algo(**{"n_components": n_components, "random_state": self.random_state})
                    data_projected = algo_trained.fit_transform(train)
                    data_dict.update({name: [data_projected, target, k]})

                    if n_components == 2:
                        self.plot_scatter(data_projected, target, f"{algo_name} reduced to {n_components} components scatter plot for {name}")

                k_means_performance_dict = self.cluster(data_dict, f"{algo_name} reduced to {n_components} components", "KMeans", k_means_performance_dict)
                em_performance_dict = self.cluster(data_dict, f"{algo_name} reduced to {n_components} components", "EM", em_performance_dict)

                self.elbow_method(data_dict, f"{algo_name} reduced to {n_components} components", "KMeans", self.max_k)
                # self.elbow_method(data_dict, f"{algo_name} reduced to {n_components} components", "EM", self.max_k)
            print(k_means_performance_dict, em_performance_dict)

    def rerun_neural_network(self):
        training_ratio_list = np.arange(0.1, 0.9, 0.05)
        neural_network_model = MLPClassifier(**{'hidden_layer_sizes': (300, 300, 300, 300, 300,), 'learning_rate_init': 0.001, 'tol': 0.006,'learning_rate': 'invscaling'})
        train, target, _ = self.data_dict["Bank_data"]

        dr_dict = {"PCA": [PCA, 39], "ICA": [ICA, 33], "Random_Projection": [RandomProjection, 45], "SequentialFeatureSelector": ['duration','cons.conf.idx','nr.employed','contact_telephone','month_jul','month_jun','month_mar','month_may','poutcome_success'], "Original": None}


        fig, axes = plt.subplots(1, 5, figsize=(20, 6))
        fig.suptitle(f"Bank Marketing Dataset", fontsize=20)
        i = 0

        for algo_name, info in dr_dict.items():
            if algo_name == "SequentialFeatureSelector":
                data_projected = train[info]
            elif algo_name != "Original":
                algo, n_components = info[0], info[1]
                algo_trained = algo(**{"n_components": n_components, "random_state": self.random_state})
                data_projected = algo_trained.fit_transform(train)
            else:
                data_projected = train

            train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(neural_network_model, data_projected, target, train_sizes=training_ratio_list, cv=5, scoring="accuracy", return_times=True)
            train_scores, test_scores = 1 - train_scores, 1 - test_scores  # make it as error rate
            print(algo_name, train_sizes, train_scores, test_scores)

            # plot graph
            train_scores_mean, train_scores_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
            test_scores_mean, test_scores_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)

            axes[i].grid()
            axes[i].set_title(f"{algo_name}")
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
        fig.savefig(os.path.join(self.graph_path, f"Bank_Marketing_learning_Curve"))
        fig.close()

    def rerun_neural_network_clustering(self):
        training_ratio_list = np.arange(0.1, 0.9, 0.05)
        neural_network_model = MLPClassifier(**{'hidden_layer_sizes': (300, 300, 300, 300, 300,), 'learning_rate_init': 0.001, 'tol': 0.006,'learning_rate': 'invscaling'})
        train, target, _ = self.data_dict["Bank_data"]

        dr_dict = {"Kmeans": [KMeans, 2], "EM": [EM, 2], "Original": None}

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f"Bank Marketing Dataset NN Learning Curve After Clustering", fontsize=20)
        i = 0

        for algo_name, info in dr_dict.items():
            if algo_name != "Original":
                algo, n_k = info[0], info[1]
                algo_trained = algo(**{"n_clusters": n_k, "random_state": self.random_state}) if algo_name == "Kmeans" else algo(**{"n_components": n_k, "random_state": self.random_state})
                data_projected = algo_trained.fit_predict(train).reshape(-1, 1)
                new_train = copy.deepcopy(train)
                new_train["Cluster_result"] = data_projected
            else:
                data_projected = copy.deepcopy(train)

            train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(neural_network_model, data_projected, target, train_sizes=training_ratio_list, cv=5, scoring="accuracy", return_times=True)
            train_scores, test_scores = 1 - train_scores, 1 - test_scores  # make it as error rate
            print(algo_name, train_sizes, train_scores, test_scores)

            # plot graph
            train_scores_mean, train_scores_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
            test_scores_mean, test_scores_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)

            axes[i].grid()
            axes[i].set_title(f"{algo_name}")
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
        fig.savefig(os.path.join(self.graph_path, f"Bank_Marketing_learning_Curve after clustering"))
        plt.close()

    def roc_score_4_5(self):
        neural_network_model = MLPClassifier(
            **{'hidden_layer_sizes': (300, 300, 300, 300, 300,), 'learning_rate_init': 0.001, 'tol': 0.006,
               'learning_rate': 'invscaling'})
        train, target, _ = self.data_dict["Bank_data"]
        x_train, x_test, y_train, y_test = train_test_split(train, target, train_size=0.8)

        dr_dict = {"PCA": [PCA, 39], "ICA": [ICA, 33], "Random_Projection": [RandomProjection, 45],
                   "SequentialFeatureSelector": ['duration', 'cons.conf.idx', 'nr.employed', 'contact_telephone',
                                                 'month_jul', 'month_jun', 'month_mar', 'month_may',
                                                 'poutcome_success'], "Original": None}

        for algo_name, info in dr_dict.items():
            if algo_name == "SequentialFeatureSelector":
                data_projected = x_train[info]
                data_test = x_test[info]

            elif algo_name != "Original":
                algo, n_components = info[0], info[1]
                algo_trained = algo(**{"n_components": n_components, "random_state": self.random_state})
                data_projected = algo_trained.fit_transform(x_train)
                data_test = algo_trained.transform(x_test)
            else:
                data_projected = x_train
                data_test = x_test

            start_time = time.time()
            neural_network_model.fit(data_projected, y_train)
            training_result = roc_auc_score(y_train, neural_network_model.predict_proba(data_projected)[:, 1])
            testing_result = roc_auc_score(y_test, neural_network_model.predict_proba(data_test)[:, 1])
            end_time = time.time()
            print(algo_name, training_result, testing_result, end_time-start_time)


        dr_dict = {"Kmeans": [KMeans, 2], "EM": [EM, 2], "Original": None}
        for algo_name, info in dr_dict.items():
            if algo_name != "Original":
                algo, n_k = info[0], info[1]
                algo_trained = algo(**{"n_clusters": n_k, "random_state": self.random_state}) if algo_name == "Kmeans" else algo(**{"n_components": n_k, "random_state": self.random_state})
                data_projected = algo_trained.fit_predict(x_train).reshape(-1, 1)
                new_train = copy.deepcopy(x_train)
                new_train["Cluster_result"] = data_projected

                test_projected = algo_trained.predict(x_test)
                new_test = copy.deepcopy(x_test)
                new_test["Cluster_result"] = test_projected
            else:
                new_train = copy.deepcopy(x_train)
                new_test = copy.deepcopy(x_test)

            start_time = time.time()
            neural_network_model.fit(new_train, y_train)
            training_result = roc_auc_score(y_train, neural_network_model.predict_proba(new_train)[:, 1])
            testing_result = roc_auc_score(y_test, neural_network_model.predict_proba(new_test)[:, 1])
            end_time = time.time()
            print(algo_name, training_result, testing_result, end_time-start_time)

    def plot_heat_map(self, target, labels, name):
        mat = confusion_matrix(target, labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=list(set(target.values)),
                    yticklabels=list(set(target.values)),
                    cmap='viridis_r')
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.savefig(os.path.join(self.graph_path, name))
        plt.close()

    def plot_scatter(self, projection, target, name):
        plt.figure(figsize=(10, 6))
        plt.scatter(projection[:, 0], projection[:, 1], c=target, cmap="Paired")
        plt.title(name)
        plt.colorbar()
        plt.savefig(os.path.join(self.graph_path, name))
        plt.close()


if __name__ == "__main__":
    UnsupervisedLearning()