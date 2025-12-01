import matplotlib.pyplot as plt
import sklearn
import numpy as np
import math
from hierarchy import HierarchicalClustering, Granule, calculate_fcm_variance
from fuzzy_cmeans_lib import FuzzyCMeans
import os
import time
import skfuzzy as fuzz

# Constants
seed = 42
n_granules = 200
n_iterations = 80
repeats = 4
s = 3
folderPath = "img/iter80/"


class DataEntry(object):
    def __init__(self, data, name, dim=2):
        self.x = []
        self.y = []
        self.z = []
        data = data.splitlines()
        for line in data:
            coords = line.split()
            self.x.append(float(coords[0]))
            self.y.append(float(coords[1]))
            if dim > 2:
                self.z.append(float(coords[2]))
        if dim > 2:
            self.data = np.array(list(zip(self.x, self.y, self.z)))
        else:
            self.data = np.array(list(zip(self.x, self.y)))
        self.name = name
        self.length = len(self.x)
        self.clusters_number = 0
        self.granules_number = [50, 100, 200]
        self.ksi = [0.1, 0.5, 0.9]

        self.strict_number_times = []
        self.fuzzy_number_times = dict()

    def measure_strict_number_times(self):
        self.strict_number_times = []
        print("strict")
        for i in range(repeats):
            hc = HierarchicalClustering(self.clusters_number)
            start = time.time()
            hc.fit(self.data)
            end = time.time()
            self.strict_number_times.append((end - start) * 1000)

    def measure_fuzzy_number_times(self):
        self.fuzzy_number_times = dict()
        for n in self.granules_number:
            print(n)
            for i in range(repeats):
                fcm = FuzzyCMeans(n_clusters=n, random_state=seed, max_iter=n_iterations)
                hc = HierarchicalClustering(n_clusters=self.clusters_number)
                granules = []

                start = time.time()
                fcm.fit(self.data)
                fuzziness = calculate_fcm_variance(fcm)
                for i in range(n_granules):
                    granules.append(Granule(fcm.cluster_centers_[i], fuzziness[i]))
                hc.fuzzy_fit(granules, 0.5)
                end = time.time()
                self.fuzzy_number_times[n] = (end - start) * 1000

    def measure_times(self):
        self.measure_strict_number_times()
        self.measure_fuzzy_number_times()
        print(np.mean(self.strict_number_times))
        for n in self.granules_number:
            print(np.mean(self.fuzzy_number_times[n]))

    def fit_plot_fuzzy_labels(self, relation_type):
        print(self.name)
        labels = []
        centers = []
        for n in self.granules_number:
            fcm = FuzzyCMeans(n_clusters=n, random_state=seed, max_iter=n_iterations)
            granules = []

            fcm.fit(self.data)
            centers.append(fcm.cluster_centers_)
            fuzziness = calculate_fcm_variance(fcm)
            for i in range(n):
                granules.append(Granule(fcm.cluster_centers_[i], fuzziness[i]))

            labels_n = []
            for k in self.ksi:
                hc = HierarchicalClustering(n_clusters=self.clusters_number)
                hc.fuzzy_fit(granules, k, relation_type)
                labels_n.append(hc.labels)
            labels.append(labels_n)

        if len(self.z) == 0:
            fig, ax = plt.subplots(len(self.granules_number), len(self.ksi))
            fig.set_figheight(10)
            fig.set_figwidth(12)

            for i in range(len(self.granules_number)):
                for j in range(len(self.ksi)):
                    ax[i, j].scatter(self.data[:, 0], self.data[:, 1], c='lightgray')
                    ax[i, j].scatter(centers[i][:, 0], centers[i][:, 1], c=labels[i][j], cmap='cool')
                    ax[i, j].set_title(str(self.granules_number[i]) + " granules, ksi = " + str(self.ksi[j]))
            # plt.savefig(folderPath + self.name + relation_type + ".pdf")
            plt.show()
            plt.close()
        else:
            fig = plt.figure(figsize=(12, 10))
            for i in range(len(self.granules_number)):
                for j in range(len(self.ksi)):
                    ax = fig.add_subplot(len(self.granules_number), len(self.ksi), i*len(self.granules_number) + j + 1, projection='3d')
                    # ax.scatter3D(self.data[:, 0], self.data[:, 1], self.data[:, 2], c='lightgray')
                    ax.scatter3D(centers[i][:, 0], centers[i][:, 1], centers[i][:, 2], c=labels[i][j], cmap='cool')
                    ax.set_title(str(self.granules_number[i]) + " granules, ksi = " + str(self.ksi[j]))
            # plt.savefig(folderPath + self.name + relation_type + ".pdf")
            plt.show()
            plt.close()


fullData = {}
names = []

# Values picked by visually deciding the best number of clusters for a given shape
num_of_clusters = {"blobs": 3, "circles": 2, "corners": 4, "crescents": 2, "laguna": 3, "spheres": 2}
names_roots = ["blobs", "circles", "corners", "crescents", "laguna", "spheres"]
# blobs10000 -> 7 minutes of clustering
# data_size_presets = [1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]
data_size_presets = [1000, 2000, 5000]

for folder in os.scandir("dane"):
    for file in os.scandir(folder.path):
        with open(file.path) as f:
            # Assumption: all folders contain only files with .data extension
            names.append(file.name[:-5])
            if "spheres" in file.name:
                fullData[file.name[:-5]] = DataEntry(f.read(), file.name[:-5], 3)
            else:
                fullData[file.name[:-5]] = DataEntry(f.read(), file.name[:-5])
            for key, value in num_of_clusters.items():
                if key in file.name:
                    fullData[file.name[:-5]].clusters_number = value
                    break

# fullData['blobs2000'].fit_plot_fuzzy_labels('t')

# data = fullData['blobs1000'].data
# cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(np.transpose(data), 100, 1.2, error=1e-4, maxiter=150, init=None)
# plt.figure()
# plt.scatter(data[:, 0], data[:, 1], c='lightgray')
# print(u)
# plt.scatter(cntr[:, 0], cntr[:, 1], c=np.argmax(u, axis=1), cmap='cool')
# plt.show()

# for name in names:
#     if 'spheres' in name:
#         fullData[name].fit_plot_fuzzy_labels('t')
#         fullData[name].fit_plot_fuzzy_labels('e')
#         fullData[name].fit_plot_fuzzy_labels('g')
# for nr in names_roots:
#     for dsp in data_size_presets:
#         print("Measure ", nr, str(dsp))
#         fullData[nr + str(dsp)].measure_times()
#
# regular_avg = {"blobs": np.empty(shape=s), "circles": np.empty(shape=s), "corners": np.empty(shape=s), "crescents": np.empty(shape=s), "laguna": np.empty(shape=s), "spheres": np.empty(shape=s)}
# granule50_avg = {"blobs": np.empty(shape=s), "circles": np.empty(shape=s), "corners": np.empty(shape=s), "crescents": np.empty(shape=s), "laguna": np.empty(shape=s), "spheres": np.empty(shape=s)}
# granule100_avg = {"blobs": np.empty(shape=s), "circles": np.empty(shape=s), "corners": np.empty(shape=s), "crescents": np.empty(shape=s), "laguna": np.empty(shape=s), "spheres": np.empty(shape=s)}
# granule200_avg = {"blobs": np.empty(shape=s), "circles": np.empty(shape=s), "corners": np.empty(shape=s), "crescents": np.empty(shape=s), "laguna": np.empty(shape=s), "spheres": np.empty(shape=s)}
#
# for nr in names_roots:
#     for i in range(len(data_size_presets)):
#         regular_avg[nr][i] = np.mean(fullData[nr + str(data_size_presets[i])].strict_number_times)
#         granule50_avg[nr][i] = np.mean(fullData[nr + str(data_size_presets[i])].fuzzy_number_times[50])
#         granule100_avg[nr][i] = np.mean(fullData[nr + str(data_size_presets[i])].fuzzy_number_times[100])
#         granule200_avg[nr][i] = np.mean(fullData[nr + str(data_size_presets[i])].fuzzy_number_times[200])
#
# for name in names_roots:
#     plt.figure()
#     plt.plot(data_size_presets, regular_avg[name])
#     plt.plot(data_size_presets, granule50_avg[name])
#     plt.plot(data_size_presets, granule100_avg[name])
#     plt.plot(data_size_presets, granule200_avg[name])
#     plt.xlabel("Number of data")
#     plt.ylabel("Time [ms]")
#     plt.title("Clustering times of " + name + " data")
#     legend = plt.legend(["Hierarchical clustering of non-granulated data", "Hierarchical clustering of 50 granules", "Hierarchical clustering of 100 granules", "Hierarchical clustering of 200 granules"], bbox_to_anchor =(0.5,-0.27), loc='lower center')
#     plt.savefig(folderPath + name + "_times.pdf", bbox_extra_artists=[legend], bbox_inches='tight')

# Generate data
data, labels = sklearn.datasets.make_blobs(n_samples=1000, centers=4, cluster_std=[0.5, 2, 1.5, 1], random_state=seed)
# data, labels = sklearn.datasets.make_blobs(n_samples=1000, centers=4, random_state=100)
counts = { 0: 0, 1: 0, 2: 0, 3: 0}
for label in labels:
    counts[label] += 1
print(counts)

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
hc.fuzzy_fit(granules, 0.8)

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
