# data sources https://snap.stanford.edu/data/loc-gowalla.html https://snap.stanford.edu/data/loc-brightkite.html
from data_entry import DataEntry
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from hierarchy import HierarchicalClustering, calculate_fcm_variance, Granule
from fuzzy_cmeans import FuzzyCMeans


def extractCoordinates(name):
    lines = None
    with open("datasets/" + name + "_totalCheckins.txt") as f:
        lines = f.read()
        lines = lines.splitlines()

    shortened = []
    for line in lines:
        s = line.split()
        if len(s) > 3:
            shortened.append([s[2], s[3]])

    with open("datasets/" + name+ "_coordinates.txt", "w") as f:
        for line in shortened:
            f.write('\t'.join(line) + '\n')


# extractCoordinates("Brightkite")
with open("datasets/Brightkite_coordinates.txt") as f:
    long_data = DataEntry(f.read(), "brightkite", 2)

np.random.seed(42)
data1000 = DataEntry(long_data.data[np.random.choice(long_data.data.shape[0], 100000, replace=False), :], "brightkite1000", 2)
data1000.clusters_number = 3
ac = sklearn.cluster.AgglomerativeClustering(n_clusters=3, linkage='single')
ac.fit(data1000.data)
plt.figure()
plt.scatter(data1000.data[:, 0], data1000.data[:, 1], c=ac.labels_)
plt.show()
fcm = FuzzyCMeans(n_clusters=50, random_state=42)
fcm.fit(data1000.data)
fuzziness = calculate_fcm_variance(fcm)
# print(fuzziness)
granules = []
for i in range(50):
    granules.append(Granule(fcm.cluster_centers_[i], fuzziness[i]))
hc = HierarchicalClustering(n_clusters=3)
hc.fuzzy_fit(granules, 0.1, 't')
# Visualize data
plt.figure()
plt.scatter(data1000.data[:, 0], data1000.data[:, 1], c='lightgray')
plt.scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], c=hc.labels)

t = np.linspace(0, 2 * np.pi)
for clust in range(fcm.n_clusters):
    plt.plot(fcm.cluster_centers_[clust, 0] + 2 * fuzziness[clust][0] * np.cos(t),
             fcm.cluster_centers_[clust, 1] + 2 * fuzziness[clust][1] * np.sin(t),
             color='crimson')

plt.show()
# result = data1000.measure_accuracy()
# print(result.to_string())
# data1000.fit_plot_fuzzy_labels('t')

