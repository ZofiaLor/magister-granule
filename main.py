import matplotlib.pyplot as plt
import sklearn
import numpy as np
from hierarchy import HierarchicalClustering
from fuzzy_cmeans import FuzzyCMeans

# Constants
seed = 42

# Generate data
data, labels = sklearn.datasets.make_blobs(n_samples=10000, centers=4, cluster_std=[0.5, 2, 1.5, 1], random_state=seed)

# A small data set for manual verification from https://www.w3schools.com/python/python_ml_hierarchial_clustering.asp
data_small = np.array([[4, 21], [5, 19], [10, 24], [4, 17], [3, 16], [11, 25], [14, 24], [6, 22], [10, 21], [12, 21]])

hc = HierarchicalClustering()
hc.fit(data_small)
print(hc.labels)

# Use fuzzy c-means
# fcm = FuzzyCMeans(n_clusters=100, random_state=seed)
# fcm.fit(data, labels)

# Visualize data
# plt.figure()
# plt.scatter(data[:, 0], data[:, 1], c=labels)
# plt.scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], c='deeppink')
# plt.show()

plt.figure()
plt.scatter(data_small[:, 0], data_small[:, 1], c=hc.labels)
plt.show()
