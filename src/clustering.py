from itertools import permutations
import numpy as np


class Clustering:
    def __init__(self, k):
        self._k = k
        self._centroids = None
        self._distortion = None
        self._y_hat = None
        self._time = -1
    
    @property
    def time(self):
        return self._time

    def fit(self, X):
        raise NotImplementedError

    def accuracy(self, Y):
        if self._y_hat is None:
            raise Exception('Must run fit first.')
        n = Y.shape
        accuracy = np.zeros(self._k)
        perms = []
        for p in permutations(np.arange(1, self._k + 1)):
            P = dict()
            for i in range(self._k):
                P[i] = p[i]
            perms.append(P)
        accuracy = np.zeros(len(perms))
        for i in range(len(perms)):
            yy = self._y_hat.copy()
            for k, v in perms[i].items():
                yy[self._y_hat == k] = v
            accuracy[i] = (Y == yy).sum() / n * 100
        return accuracy.max()
