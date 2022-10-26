import numpy as np

from .CDataPerturb import CDataPerturb


class CDataPerturbGaussian(CDataPerturb):
    def __init__(self, min_value=0, max_value=255, sigma=1):
        self.min_value = min_value
        self.max_value = max_value
        self.sigma = sigma

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    @property
    def sigma(self):
        return self._sigma

    @min_value.setter
    def min_value(self, value):
        if value < 0:
            value = 0

        self._min_value = value

    @max_value.setter
    def max_value(self, value):
        if value > 255:
            value = 0

        self._max_value = value

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

    def data_perturbation(self, x):
        z = self.sigma * np.random.randn(x.size)  # sampling from N(0, sigma)
        x += z
        x[x < self.min_value] = self.min_value
        x[x > self.max_value] = self.max_value

        return x
