from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

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
# sns.set(style="ticks")


digits = load_digits()

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)

# fig, ax = plt.subplots(2, 5, figsize=(8, 3))
# centers = kmeans.cluster_centers_.reshape(10, 8, 8)
# for axi, center in zip(ax.flat, centers):
#     axi.set(xticks=[], yticks=[])
#     axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

from scipy.stats import mode

labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

from sklearn.metrics import accuracy_score
accuracy_score(digits.target, labels)

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.show()
plt.xlabel('true label')
plt.ylabel('predicted label')