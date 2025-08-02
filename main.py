import matplotlib.pyplot as plt
import sklearn
import numpy as np
import math

import hierarchy
from hierarchy import HierarchicalClustering
from fuzzy_cmeans import FuzzyCMeans

# Constants
seed = 42


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


def fuzzy_distance(centers, variances):
    n_granules = centers.shape[0]
    distances = np.zeros(shape=(n_granules, n_granules))
    for row in range(n_granules):
        for col in range(row):
            # Use the variances to compute a fuzzy distance between points
            # The coordinates of the edges is a sum of the center and the variance in a given dimension
            # The distances between the centers and the variance edges are summed
            # TODO this does not make sense with the ellipse visualization, might not be correct
            euclidean_dist = math.dist(centers[row, :], centers[col, :])
            fuzzy_dist = math.dist(centers[row, :] + variances[row, :], centers[col, :] + variances[col, :])
            distances[row, col] = euclidean_dist + fuzzy_dist
            distances[col, row] = euclidean_dist + fuzzy_dist
    return distances


# Generate data
data, labels = sklearn.datasets.make_blobs(n_samples=1000, centers=4, cluster_std=[0.5, 2, 1.5, 1], random_state=seed)

# A small data set for manual verification from https://www.w3schools.com/python/python_ml_hierarchial_clustering.asp
data_small = np.array([[4, 21], [5, 19], [10, 24], [4, 17], [3, 16], [11, 25], [14, 24], [6, 22], [10, 21], [12, 21]])

ac = sklearn.cluster.AgglomerativeClustering(n_clusters=4, linkage='single')
ac.fit(data)

# Use fuzzy c-means
fcm = FuzzyCMeans(n_clusters=50, random_state=seed)
fcm.fit(data)

fuzziness = calculate_fcm_variance(fcm)
dist = fuzzy_distance(fcm.cluster_centers_, fuzziness)
acf = sklearn.cluster.AgglomerativeClustering(n_clusters=4, metric='precomputed', linkage='single')
acf.fit(dist)

# Visualize data
plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=ac.labels_)
plt.scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], c=acf.labels_, cmap='cool')

t = np.linspace(0, 2 * np.pi)
for clust in range(fcm.n_clusters):
    plt.plot(fcm.cluster_centers_[clust, 0] + 2 * fuzziness[clust][0] * np.cos(t),
             fcm.cluster_centers_[clust, 1] + 2 * fuzziness[clust][1] * np.sin(t),
             color='crimson')

plt.show()

# plt.figure()
# plt.scatter(data_small[:, 0], data_small[:, 1], c=hc.labels)
# plt.show()

hc = hierarchy.HierarchicalClustering(n_clusters=4)
hc.fit(data)

plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=hc.labels)

plt.show()
