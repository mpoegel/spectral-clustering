# # $k$-means-based approximate spectral clustering
# This algorithm is from "[Fast Approximate Spectral Clustering](https://people.eecs.berkeley.edu/~jordan/papers/yan-huang-jordan-kdd09.pdf)" by Yan et el.

import numpy as np
import scipy as sp
import scipy.sparse.linalg
import scipy.cluster.vq
import time
from clustering import Clustering


class KASP(Clustering):
    def __init__(self, k, gamma):
        super().__init__(k)
        self._gamma = gamma

    def fit(self, X):
        start = time.time()
        n, d = X.shape
        k_prime = n // self._gamma
        # rough clustering
        centroids_prime, y_prime = sp.cluster.vq.kmeans2(X, k_prime)
        X_prime = centroids_prime
        n_prime, d_prime = X_prime.shape
        n_prime, d_prime
        # create the affinity matrix on the new points
        W = np.empty((n_prime, n_prime))
        for i in range(n_prime):
            for j in range(i, n_prime):
                val = np.linalg.norm(X_prime[i] - X_prime[j]) ** 2
                W[i, j] = val
                W[j, i] = val

        ww = W.sum(axis=0)
        D = np.diag(ww)
        D_ = np.diag(1 / np.sqrt(ww))
        L = np.identity(n_prime) - D_.dot(W).dot(D_)
        # now do spectral clustering
        V, Z = sp.linalg.eigh(L, eigvals=(n_prime-2, n_prime-1))

        Z_ = sp.cluster.vq.whiten(Z)
        self._centroids, y_hat_prime = sp.cluster.vq.kmeans2(Z_, self._k)

        self._y_hat = np.empty(n, dtype=int)
        for i in range(n):
            self._y_hat[i] = int(y_hat_prime[ int(y_prime[i]) ])
        self._time = time.time() - start


if __name__ == '__main__':
    A = np.loadtxt('data/processed/usps.csv', delimiter=',')
    inds = A[:, -1] < 3
    X = A[inds, :-2]
    Y = A[inds, -1].astype(int)
    k = 2
    gamma = 2

    model = KASP(k, gamma)
    model.fit(X)
    print(model._centroids)
    accuracy = model.accuracy(Y)
    print(accuracy)
