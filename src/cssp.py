# # Column Sampling Spectral Clustering (CSSP)
# This algorithm is from "Time and Space Effecient Spectral Clustering via Column Sampling" by Li et
# al.

import numpy as np
import scipy as sp
import scipy.sparse.linalg
import scipy.cluster.vq
import time
from clustering import Clustering


class CSSP(Clustering):
    def __init__(self, k, m):
        super().__init__(k)
        self._m = m

    def fit(self, X):
        start = time.time()
        n, d = X.shape
        # sample m columns uniformly at random without replacement
        inds = np.random.choice(n, self._m, replace=False)
        Z = X[inds, :]
        # calculate the gaussian kernel parameter
        mu = 0
        for i in range(self._m):
            for j in range(self._m):
                mu += np.linalg.norm(Z[i] - Z[j]) ** 2
        mu /= (self._m ** 2)
        mu = 1 / mu
        # created a sample of the affinity matrix
        A_11 = np.empty((self._m, self._m))
        for i in range(self._m):
            for j in range(i, self._m):
                val = np.e ** (-mu * np.linalg.norm(Z[i] - Z[j]) ** 2)
                A_11[i, j] = val
                A_11[j, i] = val

        ww = A_11.dot(np.ones(self._m))
        D_star = np.diag(ww)
        D_star_ = np.diag(1 / np.sqrt(ww))
        M_star = D_star_.dot(A_11).dot(D_star_)
        # find the eigendecomposition of M_star
        M_star = sp.cluster.vq.whiten(M_star)
        Lam, V = sp.sparse.linalg.eigsh(M_star, k=self._k, which='LM')

        Lam = np.diag(Lam)
        B = D_star_.dot(V).dot(np.linalg.inv(Lam))
        # create another affinity matrix row by row
        Q = np.empty((n, self._k))
        for i in range(n):
            a = np.array([np.linalg.norm(X[i] - Z[j]) for j in range(self._m)])
            Q[i] = a.dot(B)

        dd = Q.dot(Lam).dot(Q.T).dot(np.ones(n))
        D_hat = np.diag(dd)
        U = np.diag(1 / np.sqrt(dd)).dot(Q)
        # orthogonalize U
        P = U.T.dot(U)
        Sig, Vp = sp.linalg.eigh(P)
        Sig_ = np.diag(np.sqrt(Sig))
        B = Sig_.dot(Vp.T).dot(Lam).dot(Vp).dot(Sig_)
        Lam_tilde, V_tilde = sp.linalg.eigh(B)
        self._U = U.dot(Vp).dot(np.diag(1 / np.sqrt(Sig))).dot(V_tilde)
        # finally we have U as the approximate eigenvectors which we use to cluster
        self._centroids, self._distortion = sp.cluster.vq.kmeans(self._U, self._k)
        # calculate y_hat
        self._y_hat = np.zeros(n, dtype=int)
        for i in range(n):
            dists = np.array([np.linalg.norm(self._U[i] - self._centroids[c])
                              for c in range(self._k)])
            self._y_hat[i] = np.argmin(dists)
        self._time = time.time() - start


if __name__ == '__main__':
    A = np.loadtxt('data/processed/usps.csv', delimiter=',')
    inds = A[:, -1] < 3
    X = A[inds, :-2]
    Y = A[inds, -1].astype(int)
    k = 2
    m = 1000

    cssp = CSSP(k, m)
    cssp.fit(X)
    print(cssp._centroids)
    accuracy = cssp.accuracy(Y)
    print(accuracy)
