import matplotlib.pyplot as plt
import sklearn
from fuzzy_cmeans import FuzzyCMeans

# Constants
seed = 42

# Generate data
data, labels = sklearn.datasets.make_blobs(n_samples=10000, centers=4, cluster_std=[0.5, 2, 1.5, 1], random_state=seed)

# Use fuzzy c-means
fcm = FuzzyCMeans(n_clusters=100, random_state=seed)
fcm.fit(data, labels)

# Visualize data
plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], c='deeppink')
plt.show()
