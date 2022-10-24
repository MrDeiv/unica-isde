import numpy as np

from .CDataPerturb import CDataPerturb


class CDataPerturbGaussian(CDataPerturb):
    def __init__(self, min_value=0, max_value=255, sigma=40):
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
        for elem in x:
            elem = self.sigma * np.random.randn()

        return x
