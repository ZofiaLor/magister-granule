import math
import numpy as np
import sklearn


class HierarchicalClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.distances = None
        self.labels = None

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
