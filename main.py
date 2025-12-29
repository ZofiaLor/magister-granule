import matplotlib.pyplot as plt
import sklearn
import numpy as np
import math
from hierarchy import HierarchicalClustering, Granule, calculate_fcm_variance
from fuzzy_cmeans import FuzzyCMeans
import os
import time
import skfuzzy as fuzz
import pandas

# Constants
seed = 42
n_granules = 50
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

    def calculate_accuracy(self, fcm, hc, nonfuzzy_labels):
        granule_membership = []
        granule_member_labels = []
        for sample in fcm.U_:
            membership = int(np.argmax(sample))
            granule_membership.append(membership)
            granule_member_labels.append(int(hc.labels[membership]))

        l_matrix = np.zeros(shape=(self.clusters_number, self.clusters_number))
        for d in range(len(granule_member_labels)):
            l_matrix[nonfuzzy_labels[d], granule_member_labels[d]] = l_matrix[nonfuzzy_labels[d], granule_member_labels[
                d]] + 1

        for i in range(self.clusters_number - 1):
            (row, col) = np.unravel_index(np.argmax(l_matrix[i:, i:]), (self.clusters_number - i, self.clusters_number - i))
            row = row + i
            col = col + i
            l_matrix[[i, row], :] = l_matrix[[row, i], :]
            l_matrix[:, [i, col]] = l_matrix[:, [col, i]]

        sum_diagonal = 0
        for i in range(self.clusters_number):
            sum_diagonal = sum_diagonal + l_matrix[i, i]
        return sum_diagonal / len(granule_member_labels)

    def measure_accuracy(self):
        result_list = []
        ac = sklearn.cluster.AgglomerativeClustering(n_clusters=self.clusters_number, linkage='single')
        ac.fit(self.data)
        for n in self.granules_number:
            fcm = FuzzyCMeans(n_clusters=n, random_state=seed, max_iter=n_iterations)
            hc = HierarchicalClustering(n_clusters=self.clusters_number)
            granules = []

            fcm.fit(self.data)
            fuzziness = calculate_fcm_variance(fcm)
            for i in range(n):
                granules.append(Granule(fcm.cluster_centers_[i], fuzziness[i]))
            for relation_type in ['t', 'e', 'g']:
                for k100 in range(5, 100, 5):
                    k = k100 / 100
                    print("n " + str(n) + " type " + relation_type + " ksi " + str(k))
                    hc.fuzzy_fit(granules, k, relation_type)
                    acc = self.calculate_accuracy(fcm, hc, ac.labels_)
                    results = {"data size": self.length, "granules number": n, "relation type": relation_type, "ksi": k, "accuracy": acc}
                    result_list.append(results)
        return pandas.DataFrame(result_list)

    def fit_plot_fuzzy_labels(self, relation_type):
        print(self.name)
        labels = []
        centers = []
        fuzz = []
        for n in self.granules_number:
            fcm = FuzzyCMeans(n_clusters=n, random_state=seed, max_iter=n_iterations)
            granules = []

            fcm.fit(self.data)
            centers.append(fcm.cluster_centers_)
            fuzziness = calculate_fcm_variance(fcm)
            for i in range(n):
                granules.append(Granule(fcm.cluster_centers_[i], fuzziness[i]))
            fuzz.append(fuzziness)

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
                    t = np.linspace(0, 2 * np.pi)
                    for clust in range(self.granules_number[i]):
                        ax[i, j].plot(centers[i][clust, 0] + 2 * fuzz[i][clust][0] * np.cos(t),
                                      centers[i][clust, 1] + 2 * fuzz[i][clust][1] * np.sin(t),
                                      color='crimson', alpha=0.5)
            plt.savefig(folderPath + self.name + relation_type + ".pdf")
            # plt.show()
            plt.close()
        else:
            fig = plt.figure(figsize=(12, 10))
            for i in range(len(self.granules_number)):
                for j in range(len(self.ksi)):
                    t = np.linspace(0, 2 * np.pi, num=10)
                    p = np.linspace(0, np.pi, num=10)
                    t, p = np.meshgrid(t, p)
                    ax = fig.add_subplot(len(self.granules_number), len(self.ksi),
                                         i * len(self.granules_number) + j + 1, projection='3d')
                    # ax.scatter3D(self.data[:, 0], self.data[:, 1], self.data[:, 2], c='lightgray')
                    ax.scatter3D(centers[i][:, 0], centers[i][:, 1], centers[i][:, 2], c=labels[i][j], cmap='cool')
                    ax.set_title(str(self.granules_number[i]) + " granules, ksi = " + str(self.ksi[j]))
                    for clust in range(self.granules_number[i]):
                        ax.plot_surface(centers[i][clust, 0] + 2 * fuzz[i][clust][0] * np.sin(p) * np.cos(t),
                                        centers[i][clust, 1] + 2 * fuzz[i][clust][1] * np.sin(p) * np.sin(t),
                                        centers[i][clust, 2] + 2 * fuzz[i][clust][2] * np.cos(p),
                                        color='crimson', alpha=0.5)
            plt.savefig(folderPath + self.name + relation_type + ".pdf")
            # plt.show()
            plt.close()


fullData = {}
names = []

# Values picked by visually deciding the best number of clusters for a given shape
num_of_clusters = {"blobs": 3, "circles": 2, "corners": 4, "crescents": 2, "laguna": 3, "spheres": 2}
names_roots = ["blobs", "circles", "corners", "crescents", "laguna", "spheres"]
# blobs10000 -> 7 minutes of clustering
data_size_presets = [1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]
# data_size_presets = [1000, 2000, 5000]

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

# fullData['spheres2000'].fit_plot_fuzzy_labels('t')
# fullData['corners1000'].measure_accuracy()
for root in names_roots:
    accuracy_results = pandas.DataFrame()
    for data_size in data_size_presets:
        print(root + str(data_size))
        result = fullData[root + str(data_size)].measure_accuracy()
        accuracy_results = pandas.concat([accuracy_results, result], ignore_index=True)
    accuracy_results.to_csv("wyniki/" + root + "_accuracy.csv")



# data = fullData['blobs1000'].data
# cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(np.transpose(data), 100, 1.2, error=1e-4, maxiter=150, init=None)
# plt.figure()
# plt.scatter(data[:, 0], data[:, 1], c='lightgray')
# print(u)
# plt.scatter(cntr[:, 0], cntr[:, 1], c=np.argmax(u, axis=1), cmap='cool')
# plt.show()

# for name in names:
#     fullData[name].fit_plot_fuzzy_labels('t')
#     fullData[name].fit_plot_fuzzy_labels('e')
#     fullData[name].fit_plot_fuzzy_labels('g')
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

# A small data set for manual verification from https://www.w3schools.com/python/python_ml_hierarchial_clustering.asp
# data_small = np.array([[4, 21], [5, 19], [10, 24], [4, 17], [3, 16], [11, 25], [14, 24], [6, 22], [10, 21], [12, 21]])
# data_small, labels = sklearn.datasets.make_blobs(n_samples=50, centers=4)
# ac = sklearn.cluster.AgglomerativeClustering(n_clusters=4, linkage='single')
# ac.fit(data_small)

# ac = sklearn.cluster.AgglomerativeClustering(n_clusters=4, linkage='single')
# ac.fit(data)
tested_data = 'corners1000'
shape = 4
# # Use fuzzy c-means
fcm = FuzzyCMeans(n_clusters=n_granules, random_state=seed)
fcm.fit(fullData[tested_data].data)

fuzziness = calculate_fcm_variance(fcm)
# print(fuzziness)
granules = []
for i in range(n_granules):
    granules.append(Granule(fcm.cluster_centers_[i], fuzziness[i]))
hc = HierarchicalClustering(n_clusters=shape)
# hc.fit(fcm.cluster_centers_)
hc.fit(fullData[tested_data].data)
nonfuzzy_labels = list(hc.labels)
print("not fuzzy")
print(nonfuzzy_labels)
plt.figure()
plt.scatter(fullData[tested_data].data[:, 0], fullData[tested_data].data[:, 1], c=hc.labels)
plt.show()

hc.fuzzy_fit(granules, 0.95)
print("fuzzy")
print(hc.labels)
granule_membership = []
granule_member_labels = []
for sample in fcm.U_:
    membership = int(np.argmax(sample))
    granule_membership.append(membership)
    granule_member_labels.append(int(hc.labels[membership]))
print(granule_membership)
print(granule_member_labels)

l_matrix = np.zeros(shape=(shape, shape))
for d in range(len(granule_member_labels)):
    l_matrix[nonfuzzy_labels[d], granule_member_labels[d]] = l_matrix[nonfuzzy_labels[d], granule_member_labels[d]] + 1
print(l_matrix)

for i in range(shape - 1):
    print(l_matrix[i:, i:])
    (row, col) = np.unravel_index(np.argmax(l_matrix[i:, i:]), (shape - i, shape - i))
    row = row + i
    col = col + i
    print(row, " ", col)
    l_matrix[[i, row], :] = l_matrix[[row, i], :]
    l_matrix[:, [i, col]] = l_matrix[:, [col, i]]
    print(l_matrix)

print(l_matrix)

sum_diagonal = 0
for i in range(shape):
    sum_diagonal = sum_diagonal + l_matrix[i, i]
print("accuracy: ", sum_diagonal / len(granule_member_labels))

# Visualize data
plt.figure()
plt.scatter(fullData[tested_data].data[:, 0], fullData[tested_data].data[:, 1], c='lightgray')
plt.scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], c=hc.labels)

t = np.linspace(0, 2 * np.pi)
for clust in range(fcm.n_clusters):
    plt.plot(fcm.cluster_centers_[clust, 0] + 2 * fuzziness[clust][0] * np.cos(t),
             fcm.cluster_centers_[clust, 1] + 2 * fuzziness[clust][1] * np.sin(t),
             color='crimson')

plt.show()
