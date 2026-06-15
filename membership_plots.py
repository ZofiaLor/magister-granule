import os
import matplotlib.pyplot as plt
from data_entry import DataEntry
from hierarchy import HierarchicalClustering, Granule, calculate_fcm_variance, FuzzyNumber
from fuzzy_cmeans import FuzzyCMeans

num_of_clusters = {"blobs": 3, "circles": 2, "corners": 4, "crescents": 2, "laguna": 3, "spheres": 2}
names_roots = ["blobs", "circles", "corners", "crescents", "laguna", "spheres"]
seed = 42
n_iterations = 80

fullData = {}
names = []


def plot_membership(data: DataEntry, n_granules):
    test_data = data.data
    n_clusters = data.clusters_number
    size = data.length
    fcm = FuzzyCMeans(n_clusters=n_granules, random_state=seed, max_iter=n_iterations)
    hc = HierarchicalClustering(n_clusters=n_clusters)
    granules = []
    fcm.fit(test_data)
    fuzziness = calculate_fcm_variance(fcm)
    for i in range(n_granules):
        granules.append(Granule(fcm.cluster_centers_[i], fuzziness[i]))
    hc.fuzzy_fit(granules, 0.5, 'e', linkage="single")

    member_cluster = []
    for i in range(n_clusters):
        member_cluster.append([])

    for i in range(size):
        m = []
        for j in range(n_clusters):
            m.append(0)
        for j in range(len(granules)):
            m[hc.labels[j]] += fcm.U_[i][j]
        for j in range(n_clusters):
            member_cluster[j].append(m[j])
    if n_clusters > 3:
        fig, ax = plt.subplots(2, n_clusters // 2)
        fig.set_size_inches(2 * n_clusters, 8)
        for i in range(n_clusters):
            ax[i // 2, i % 2].scatter(test_data[:, 0], test_data[:, 1], c=member_cluster[i], cmap="Blues")
            ax[i // 2, i % 2].set_title("Klaster " + str(i))
            ax[i // 2, i % 2].set_box_aspect(1)
        plt.show()
    else:
        fig, ax = plt.subplots(1, n_clusters)
        fig.set_size_inches(4 * n_clusters, 4)
        for i in range(n_clusters):
            ax[i].scatter(test_data[:, 0], test_data[:, 1], c=member_cluster[i], cmap="Blues")
            ax[i].set_title("Klaster " + str(i))
            ax[i].set_box_aspect(1)
        plt.show()


for folder in os.scandir("dane_labelled"):
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

plot_membership(fullData["laguna40000"], 50)
