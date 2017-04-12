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
        for i in range(self._k):
            self._y_hat = (self._y_hat + i) % (self._k + 1)
            self._y_hat[self._y_hat == 0] = 1
            accuracy[i] = (Y == self._y_hat).sum() / n * 100
        return accuracy.max()
