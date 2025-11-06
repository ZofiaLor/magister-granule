import matplotlib.pyplot as plt
import sklearn
import numpy as np
import math
from hierarchy import HierarchicalClustering, Granule, FuzzyNumber
from fuzzy_cmeans import FuzzyCMeans

# Constants
seed = 42
n_granules = 50


def calculate_fcm_variance(fuzz, m=2):
    n_granules = fuzz.n_clusters
    n_attributes = fuzz.X_.shape[1]
    n_samples = fuzz.X_.shape[0]
    variances = np.empty(shape=(n_granules, n_attributes))
    for c in range(n_granules):
        for a in range(n_attributes):
            numerator = sum(
                (fuzz.U_[i, c] ** m) * (fuzz.X_[i, a] - fuzz.cluster_centers_[c, a]) ** 2 for i in range(n_samples))
            denominator = sum((fuzz.U_[i, c] ** m) for i in range(n_samples))
            variances[c, a] = np.sqrt(numerator / denominator)
    return variances


# Generate data
#data, labels = sklearn.datasets.make_blobs(n_samples=1000, centers=4, cluster_std=[0.5, 2, 1.5, 1], random_state=seed)
data, labels = sklearn.datasets.make_blobs(n_samples=1000, centers=4, random_state=100)

# A small data set for manual verification from https://www.w3schools.com/python/python_ml_hierarchial_clustering.asp
# data_small = np.array([[4, 21], [5, 19], [10, 24], [4, 17], [3, 16], [11, 25], [14, 24], [6, 22], [10, 21], [12, 21]])
# data_small, labels = sklearn.datasets.make_blobs(n_samples=50, centers=4)
# ac = sklearn.cluster.AgglomerativeClustering(n_clusters=4, linkage='single')
# ac.fit(data_small)

# ac = sklearn.cluster.AgglomerativeClustering(n_clusters=4, linkage='single')
# ac.fit(data)
#
# # Use fuzzy c-means
fcm = FuzzyCMeans(n_clusters=n_granules, random_state=seed)
fcm.fit(data)

fuzziness = calculate_fcm_variance(fcm)
granules = []
for i in range(n_granules):
    granules.append(Granule(fcm.cluster_centers_[i], fuzziness[i]))
hc = HierarchicalClustering(n_clusters=4)
hc.fuzzy_fit(granules, 0.1)

# Visualize data
plt.figure()
plt.scatter(data[:, 0], data[:, 1], c='lightgray')
plt.scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], c=hc.labels, cmap='cool')

# t = np.linspace(0, 2 * np.pi)
# for clust in range(fcm.n_clusters):
#     plt.plot(fcm.cluster_centers_[clust, 0] + 2 * fuzziness[clust][0] * np.cos(t),
#              fcm.cluster_centers_[clust, 1] + 2 * fuzziness[clust][1] * np.sin(t),
#              color='crimson')

plt.show()
