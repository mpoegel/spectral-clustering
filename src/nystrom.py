# # Standard Nystrom-based Spectral Clustering
# This algorithm is from "[Fast Spectral Clustering via the Nystrom Method](http://www.cs.columbia.edu/~jebara/papers/ALT2013FSCVTNM.pdf)" by Choromanska et el.

import numpy as np
import scipy as sp
import scipy.sparse.linalg
import scipy.cluster.vq
import time
from clustering import Clustering


class Nystrom(Clustering):
    def __init__(self, k, m):
        super().__init__(k)
        self._m = m

    def fit(self, X):
        start = time.time()
        n, d = X.shape
        m = self._m
        # create a permutation matrix to shuffle the rows of X (and later Y)
        self._P = np.identity(n)
        np.random.shuffle(self._P)
        X = X.copy()
        X = self._P.dot(X)
        inds = np.arange(m)
        inv_inds = np.array([a for a in range(n) if a not in inds])
        # calculate the guassian kernel parameter
        mu = 0
        for i in range(m):
            for j in range(m):
                mu += np.linalg.norm(X[inds[i]] - X[inds[j]]) ** 2
        mu /= (m ** 2)
        mu = 1 / mu
        # Compute a sample of the affinity matrix
        A_11 = np.empty((m, m))
        for i in range(m):
            for j in range(i, m):
                val = np.e ** (-mu * np.linalg.norm(X[inds[i]] - X[inds[j]]) ** 2)
                A_11[i, j] = val
                A_11[j, i] = val
        # Compute another sample of the affinity matrix
        C = np.empty((n-m, m))
        for i in range(n-m):
            for j in range(m):
                val = np.e ** (-mu * np.linalg.norm(X[inv_inds[i]] - X[inds[j]]) ** 2)
                C[i, j] = val
        C = np.vstack((A_11, C))
        # Calculate the approximate degree matrix
        dd_hat = C.dot(np.linalg.inv(A_11)).dot(C.T).dot(np.ones(n))
        D_hat = np.diag(dd_hat)
        D_hat_inv = np.diag(1 / np.sqrt(dd_hat))
        D_n1 = np.diag(C.dot(np.ones(m)))
        D_11_inv = np.diag(1 / np.sqrt(A_11.dot(np.ones(m))))
        # compute the approximate Lapacian
        M_hat_n1 = D_hat_inv.dot(C).dot(D_11_inv)
        M_11 = M_hat_n1[inds, :].T
        M_21 = M_hat_n1[inv_inds, :].T
        V, U = np.linalg.eig(np.linalg.inv(M_11))
        M_11_inv = U.dot(np.diag(np.sqrt(V))).dot(U.T)
        # orthogonalize the eigenvectors of M_hat to create V
        S = M_11 + M_11_inv.dot(M_21).dot(M_21.T).dot(M_11_inv)
        U, L, T = np.linalg.svd(S)
        V = M_hat_n1.dot(M_11_inv).dot(U).dot(np.diag(1 / np.sqrt(L)))
        # cluster on the first k eigenvectors
        Z = V[:, :self._k]
        self._centroids, self._y_hat = sp.cluster.vq.kmeans2(Z, self._k)
        self._time = time.time() - start

    def accuracy(self, Y):
        Y = Y.copy()
        Y = self._P.dot(Y)
        return super().accuracy(Y)


if __name__ == '__main__':
    A = np.loadtxt('data/processed/usps.csv', delimiter=',')
    inds = A[:, -1] < 3
    X = A[inds, :-2]
    Y = A[inds, -1].astype(int)
    k = 2
    m = 300

    model = Nystrom(k, m)
    model.fit(X)
    print(model._centroids)
    accuracy = model.accuracy(Y)
    print(accuracy)
