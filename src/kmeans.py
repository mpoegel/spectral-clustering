# # $K$-means

import numpy as np
import scipy as sp
import scipy.sparse.linalg
import scipy.cluster.vq
import time
from clustering import Clustering


class KMeans(Clustering):
    def __init__(self, k):
        super().__init__(k)

    def fit(self, X):
        start = time.time()
        n, d = X.shape
        self._centroids, self._y_hat = sp.cluster.vq.kmeans2(X, self._k)
        self._time = time.time() - start


if __name__ == '__main__':
    A = np.loadtxt('data/processed/usps.csv', delimiter=',')
    inds = A[:, -1] < 3
    X = A[inds, :-2]
    Y = A[inds, -1].astype(int)
    k = 2

    model = KMeans(k)
    model.fit(X)
    print(model._centroids)
    accuracy = model.accuracy(Y)
    print(accuracy)
