import math
import numpy as np
import sklearn


class FuzzyNumber:
    def __init__(self, x, p):
        self.x = x
        self.p = p

    def __repr__(self):
        return f"{self.x}({self.p})"

    def equal(self, other):
        if self.p == 0 and other.p == 0:
            if self.x == other.x:
                return 1
            return 0
        return max(1 - (abs(self.x - other.x)/max(self.p, other.p)), 0)

    def less(self, other):
        if self.x > other.x:
            return 0
        return 1 - self.equal(other)


class Granule:
    def __init__(self, center, fuzziness):
        self.center = center
        self.fuzziness = fuzziness

    def fuzzy_distance(self, other):
        distance = 0
        for i in range(self.center.shape[0]):
            distance += (self.center[i] - other.center[i]) ** 2
        distance = math.sqrt(distance)
        fuzz = np.max([self.fuzziness, other.fuzziness])
        return FuzzyNumber(x=distance, p=fuzz)


class HierarchicalClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.distances = None
        self.labels = None

    def fuzzy_distance(self, granules):
        n_samples = np.size(granules, 0)
        self.distances = np.full((n_samples, n_samples), FuzzyNumber(np.inf, 0))
        for row in range(n_samples-1):
            for col in range(row+1, n_samples):
                self.distances[row, col] = granules[row].fuzzy_distance(granules[col])
        return self.distances
        # for i in range(n_samples):
        #     for j in range(n_samples):
        #         print(repr(self.distances[i, j]), end="\t")
        #     print()

    def fuzzy_fit(self, granules, ksi):
        self.fuzzy_distance(granules)
        n_samples = np.size(granules, 0)
        self.labels = np.arange(0, n_samples, step=1)

        for it in range(n_samples - self.n_clusters):
            min_num = FuzzyNumber(np.inf, 0)
            row = 0
            col = 0
            for i in range(n_samples - 1):
                for j in range(i + 1, n_samples):
                    if self.distances[i, j].less(min_num) > ksi:
                        min_num = self.distances[i, j]
                        row = i
                        col = j

            # Iterate over one axis, change the row values to the minimum
            for i in range(n_samples):
                if i != row:
                    min_distance = FuzzyNumber(0, 0)
                    if self.distances[row, i].less(self.distances[i, col]) > ksi:
                        min_distance = self.distances[row, i]
                    else:
                        min_distance = self.distances[i, col]
                    self.distances[i, row] = min_distance
                    self.distances[row, i] = min_distance
                self.distances[i, col] = FuzzyNumber(np.inf, 0)
                self.distances[col, i] = FuzzyNumber(np.inf, 0)

            # This does not ensure that the labels are consecutive
            for i in range(n_samples):
                if self.labels[i] == max(row, col):
                    self.labels[i] = min(row, col)

        to_map = np.unique(self.labels)
        for i in range(self.labels.size):
            self.labels[i] = np.where(to_map == self.labels[i])[0][0]


    # Calculate the distance matrix using the Euclidean metric with infinity on the diagonal
    def euclidean_distance(self, X):
        self.distances = sklearn.metrics.pairwise_distances(X)
        np.fill_diagonal(self.distances, np.inf)
        return self.distances

    # Algorithm based on https://github.com/hhundiwala/hierarchical-clustering/tree/master
    def fit(self, X):
        self.euclidean_distance(X)
        n_samples = np.size(X, 0)
        self.labels = np.arange(0, n_samples, step=1)

        for it in range(n_samples - self.n_clusters):
            min_unraveled_index = np.argmin(self.distances)
            # The matrix is symmetric so the row/col distinction is not important
            # However, for the later part of the algorithm to work, row < col
            row, col = np.unravel_index(min_unraveled_index, (n_samples, n_samples))
            if row > col:
                row, col = col, row

            # Iterate over one axis, change the row values to the minimum
            for i in range(n_samples):
                if i != row:
                    min_distance = min(self.distances[row, i], self.distances[i, col])
                    self.distances[i, row] = min_distance
                    self.distances[row, i] = min_distance
                self.distances[i, col] = np.inf
                self.distances[col, i] = np.inf

            # This does not ensure that the labels are consecutive
            for i in range(n_samples):
                if self.labels[i] == max(row, col):
                    self.labels[i] = min(row, col)

        to_map = np.unique(self.labels)
        for i in range(self.labels.size):
            self.labels[i] = np.where(to_map == self.labels[i])[0][0]
