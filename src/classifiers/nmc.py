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

    def split_data(self, x, y, tr_fraction=.5):
        n_samples = y.size
        n_tr = int(tr_fraction * n_samples)
        n_ts = n_samples - n_tr

        tr_idx = np.zeros(shape=(n_samples,))
        tr_idx[0:n_tr] = 1  # 1 = training, 0 = test

        np.random.shuffle(tr_idx)  # in place operation - it modifies the object inside

        ytr = y[tr_idx == 1]
        xtr = x[tr_idx == 1, :]

        yts = y[tr_idx == 0]
        xts = x[tr_idx == 0, :]

        return xtr, ytr, xts, yts
