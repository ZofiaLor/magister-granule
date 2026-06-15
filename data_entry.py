import matplotlib.pyplot as plt
import sklearn
import numpy as np
from hierarchy import HierarchicalClustering, Granule, calculate_fcm_variance, FuzzyNumber
from fuzzy_cmeans import FuzzyCMeans
import time
import pandas

# Constants
seed = 42
n_granules = 50
n_iterations = 80
repeats = 1
folderPath = "img/iter80/"


class DataEntry(object):
    def __init__(self, data, name, dim=2, has_labels=True):
        self.labels = []
        if isinstance(data, np.ndarray):
            # handle labels
            self.data = data
        else:
            data = data.splitlines()
            x = []
            y = []
            z = []
            for line in data:
                coords = line.split()
                x.append(float(coords[0]))
                y.append(float(coords[1]))
                if dim > 2:
                    z.append(float(coords[2]))
                if has_labels:
                    self.labels.append(int(coords[-1]))
            if dim > 2:
                self.data = np.array(list(zip(x, y, z)))
            else:
                self.data = np.array(list(zip(x, y)))
        self.dim = dim
        self.name = name
        self.length = self.data.shape[0]
        rng = np.random.default_rng(seed)
        p = rng.permutation(self.length)
        if has_labels:
            self.labels = np.array(self.labels)
            self.data, self.labels = self.data[p], self.labels[p]
        else:
            self.data = self.data[p]
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
                print((end - start) * 1000)

    def measure_times(self):
        self.measure_strict_number_times()
        self.measure_fuzzy_number_times()
        print(np.mean(self.strict_number_times))
        for n in self.granules_number:
            print(np.mean(self.fuzzy_number_times[n]))

    def calculate_confusion_matrix(self, granule_member_labels, noise):
        l_matrix = np.zeros(shape=(self.clusters_number, self.clusters_number))
        for d in range(len(granule_member_labels)):
            if granule_member_labels[d] > -1:
                l_matrix[self.labels[d], granule_member_labels[d]] = l_matrix[
                                                                         self.labels[d], granule_member_labels[d]] + 1

        for i in range(self.clusters_number - 1):
            (row, col) = np.unravel_index(np.argmax(l_matrix[i:, i:]),
                                          (self.clusters_number - i, self.clusters_number - i))
            row = row + i
            col = col + i
            l_matrix[[i, row], :] = l_matrix[[row, i], :]
            l_matrix[:, [i, col]] = l_matrix[:, [col, i]]

        sum_diagonal = 0
        recall = 0
        precision = 0
        recall_denominators = np.sum(l_matrix, axis=1).tolist()
        precision_denominators = np.sum(l_matrix, axis=0).tolist()
        for i in range(self.clusters_number):
            sum_diagonal = sum_diagonal + l_matrix[i, i]
            if recall_denominators[i] != 0:
                recall += l_matrix[i, i] / recall_denominators[i]
            if precision_denominators[i] != 0:
                precision += l_matrix[i, i] / precision_denominators[i]
        accuracy = sum_diagonal / (len(granule_member_labels) - noise)
        recall /= self.clusters_number
        precision /= self.clusters_number
        noise_percentage = noise / self.length
        return accuracy, recall, precision, noise_percentage

    def calculate_accuracy(self, fcm, hc, granules, relation_type, shape_dependent_membership):
        granule_member_labels = []
        noise = 0

        if shape_dependent_membership:
            for point in self.data:
                memberships = np.zeros(len(granules))
                for i in range(len(granules)):
                    total = granules[i].fuzzy_dims[0].equal(FuzzyNumber(point[0], 0), relation_type)
                    for j in range(1, len(point)):
                        total *= granules[i].fuzzy_dims[j].equal(FuzzyNumber(point[j], 0), relation_type)
                    memberships[i] = total
                membership = np.argmax(memberships)
                if memberships[membership] > 0:
                    granule_member_labels.append(int(hc.labels[membership]))
                else:
                    noise += 1
                    granule_member_labels.append(-1)
        else:
            for sample in fcm.U_:
                membership = int(np.argmax(sample))
                granule_member_labels.append(int(hc.labels[membership]))

        return self.calculate_confusion_matrix(granule_member_labels, noise)

    def measure_accuracy(self, shape_dependent_membership, linkage='single'):
        result_list = []
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
                    hc.fuzzy_fit(granules, k, relation_type, linkage=linkage)
                    acc, rec, pre, noi = self.calculate_accuracy(fcm, hc, granules, relation_type, shape_dependent_membership)
                    results = {"data size": self.length, "granules number": n, "relation type": relation_type, "ksi": k,
                               "accuracy": acc, "recall": rec, "precision": pre, "noise percentage": noi}
                    result_list.append(results)
        return pandas.DataFrame(result_list)

    def measure_strict_accuracy(self, linkage="single"):
        result_list = []
        ac = sklearn.cluster.AgglomerativeClustering(n_clusters=self.clusters_number, linkage=linkage)
        ac.fit(self.data)
        acc, rec, pre, noi = self.calculate_confusion_matrix(ac.labels_, 0)
        result_list.append({"data size": self.length, "granules": "none", "accuracy": acc, "recall": rec, "precision": pre})
        for n in self.granules_number:
            fcm = FuzzyCMeans(n_clusters=n, random_state=seed, max_iter=n_iterations)
            fcm.fit(self.data)
            ac.fit(fcm.cluster_centers_)
            granule_member_labels = []
            for sample in fcm.U_:
                membership = int(np.argmax(sample))
                granule_member_labels.append(int(ac.labels_[membership]))
            acc, rec, pre, noi = self.calculate_confusion_matrix(granule_member_labels, 0)
            result_list.append({"data size": self.length, "granules": str(n), "accuracy": acc, "recall": rec, "precision": pre})
        return pandas.DataFrame(result_list)

    def fit_plot_fuzzy_labels(self, relation_type, linkage="single", save_to_file=True):
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
                hc.fuzzy_fit(granules, k, relation_type, linkage=linkage)
                labels_n.append(hc.labels)
            labels.append(labels_n)

        if self.dim == 2:
            fig, ax = plt.subplots(len(self.granules_number), len(self.ksi))
            fig.set_figheight(10)
            fig.set_figwidth(12)

            for i in range(len(self.granules_number)):
                for j in range(len(self.ksi)):
                    ax[i, j].scatter(self.data[:, 0], self.data[:, 1], c='lightgray')
                    ax[i, j].scatter(centers[i][:, 0], centers[i][:, 1], c=labels[i][j])
                    ax[i, j].set_title(str(self.granules_number[i]) + " granules, ksi = " + str(self.ksi[j]))
                    t = np.linspace(0, 2 * np.pi)
                    for clust in range(self.granules_number[i]):
                        ax[i, j].plot(centers[i][clust, 0] + 2 * fuzz[i][clust][0] * np.cos(t),
                                      centers[i][clust, 1] + 2 * fuzz[i][clust][1] * np.sin(t),
                                      color='magenta', alpha=0.5)
            if save_to_file:
                plt.savefig(folderPath + self.name + relation_type + "_" + linkage[0] + ".pdf")
                plt.close()
            else:
                plt.show()
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
                    ax.scatter3D(centers[i][:, 0], centers[i][:, 1], centers[i][:, 2], c=labels[i][j])
                    ax.set_title(str(self.granules_number[i]) + " granules, ksi = " + str(self.ksi[j]))
                    for clust in range(self.granules_number[i]):
                        ax.plot_surface(centers[i][clust, 0] + 2 * fuzz[i][clust][0] * np.sin(p) * np.cos(t),
                                        centers[i][clust, 1] + 2 * fuzz[i][clust][1] * np.sin(p) * np.sin(t),
                                        centers[i][clust, 2] + 2 * fuzz[i][clust][2] * np.cos(p),
                                        color='magenta', alpha=0.5)
            if save_to_file:
                plt.savefig(folderPath + self.name + relation_type + "_" + linkage[0] + ".pdf")
                plt.close()
            else:
                plt.show()
