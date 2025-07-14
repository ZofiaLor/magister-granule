import math
import numpy as np


class HierarchicalClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.distances = None
        self.labels = None

    # Calculate the distance matrix using the Euclidean metric with infinity on the diagonal
    def euclidean_distance(self, X):
        n_samples = np.size(X, 0)
        # Initialize the distance matrix of size nxn with 0s
        self.distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                self.distances[i, j] = math.dist(X[i, :], X[j, :])
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
            # This variable assignment makes it an upper triangular matrix (row < col)
            row, col = np.unravel_index(min_unraveled_index, (n_samples, n_samples))
            print(row)
            print(col)
            print(self.distances)

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
            print(self.labels)
