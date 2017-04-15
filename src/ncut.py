# # Spectral Clustering (normalized cut)

import numpy as np
import scipy as sp
import scipy.sparse.linalg
import scipy.cluster.vq
import time
from clustering import Clustering


class NCut(Clustering):
    def __init__(self, k):
        super().__init__(k)

    def fit(self, X):
        start = time.time()
        n, d = X.shape
        # calculate the gaussian kernel parameter
        mu = 0
        for i in range(n):
            for j in range(n):
                mu += np.linalg.norm(X[i] - X[j]) ** 2
        mu /= (n ** 2)
        mu = 1 / mu
        # created the entire affinity matrix 
        W = np.empty((n, n))
        for i in range(n):
            for j in range(i,n):
                val = np.e ** (-mu * np.linalg.norm(X[i] - X[j]) ** 2)
                W[i, j] = val
                W[j, i] = val
        # compute the normalized Lapacian
        ww = W.sum(axis=0)
        D = np.diag(ww)
        D_ = np.diag(1 / np.sqrt(ww))
        L = np.identity(n) - D_.dot(W).dot(D_)
        # compute the bottom k eigenvalues of L
        V, Z = sp.linalg.eigh(L, eigvals=(0, self._k))
        # and finally cluster based on the bottom k eigenvectors
        Z_ = sp.cluster.vq.whiten(Z)
        self._centroids, self._distortion = sp.cluster.vq.kmeans(Z_, self._k)
        self._y_hat = np.zeros(n, dtype=int)
        for i in range(n):
            dists = np.array([np.linalg.norm(Z_[i] - self._centroids[c]) for c in range(self._k)])
            self._y_hat[i] = np.argmin(dists)
        self._time = time.time() - start


if __name__ == '__main__':
    A = np.loadtxt('data/processed/usps.csv', delimiter=',')
    inds = A[:, -1] < 3
    X = A[inds, :-2]
    Y = A[inds, -1].astype(int)
    k = 2

    model = NCut(k)
    model.fit(X)
    print(model._centroids)
    accuracy = model.accuracy(Y)
    print(accuracy)

