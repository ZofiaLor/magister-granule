import matplotlib.pyplot as plt
import sklearn
import numpy as np
from hierarchy import HierarchicalClustering
from fuzzy_cmeans import FuzzyCMeans

# Constants
seed = 42


def calculate_fcm_variance(fuzz, m=2):
    n_granules = fuzz.n_clusters
    n_attributes = fuzz.X_.shape[1]
    n_samples = fcm.X_.shape[0]
    variances = np.empty(shape=(n_granules, n_attributes))
    for c in range(n_granules):
        for a in range(n_attributes):
            numerator = sum((fcm.U_[i, c] ** m) * (fcm.X_[i, a] - fcm.cluster_centers_[c, a]) ** 2 for i in range(n_samples))
            denominator = sum((fcm.U_[i, c] ** m) for i in range(n_samples))
            variances[c, a] = np.sqrt(numerator / denominator)
    return variances


# Generate data
data, labels = sklearn.datasets.make_blobs(n_samples=1000, centers=4, cluster_std=[0.5, 2, 1.5, 1], random_state=seed)

# A small data set for manual verification from https://www.w3schools.com/python/python_ml_hierarchial_clustering.asp
data_small = np.array([[4, 21], [5, 19], [10, 24], [4, 17], [3, 16], [11, 25], [14, 24], [6, 22], [10, 21], [12, 21]])

ac = sklearn.cluster.AgglomerativeClustering(n_clusters=4)
ac.fit(data)

# Use fuzzy c-means
fcm = FuzzyCMeans(n_clusters=50, random_state=seed)
fcm.fit(data, labels)

fuzziness = calculate_fcm_variance(fcm)

# Visualize data
plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=ac.labels_)
plt.scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], c='deeppink')

t = np.linspace(0, 2 * np.pi)
for i in range(fcm.n_clusters):
    plt.plot(fcm.cluster_centers_[i, 0] + 2 * fuzziness[i][0] * np.cos(t),
             fcm.cluster_centers_[i, 1] + 2 * fuzziness[i][1] * np.sin(t),
             color='crimson')

plt.show()

# plt.figure()
# plt.scatter(data_small[:, 0], data_small[:, 1], c=hc.labels)
# plt.show()
