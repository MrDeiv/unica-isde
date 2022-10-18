import numpy as np
from sklearn.metrics import pairwise_distances


class NMC:
    def __init__(self):
        self._centroids = None

    @property
    def centroids(self):
        return self._centroids

    def fit(self, xtr, ytr):
        n_classes = np.unique(ytr).size
        n_features = xtr.shape[1]

        self._centroids = np.zeros(shape=(n_classes, n_features))

        for k in range(n_classes):
            xk = xtr[ytr == k, :]
            self._centroids[k, :] = np.mean(xk, axis=0)

        return self._centroids

    def predict(self, xts):
        if self._centroids is None:
            raise ValueError("Train classifier first")

        dist = pairwise_distances(xts, self._centroids)
        y_pred = np.argmin(dist, axis=1)
        return y_pred
