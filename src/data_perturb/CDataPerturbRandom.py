import numpy as np

from data_perturb import CDataPerturb

class CDataPerturbRandom(CDataPerturb):

    def __init__(self, min_value=0, max_value=255, k=100):
        self.min_value = min_value
        self.max_value = max_value
        self.k = k

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    @property
    def k(self):
        return self._k

    @min_value.setter
    def min_value(self, value):
        """

        Parameters
        ----------
        value

        Returns
        -------

        """
        if value<0:
            value = 0

        self._min_value = value

    @max_value.setter
    def max_value(self, value):
        if value > 255:
            value = 0

        self._max_value = value

    @k.setter
    def k(self, k):
        self._k = k

    def data_perturbation(self, x):
        #k values
        p_elem = np.random.choice(np.arange(0, x.size), replace=False, size=(self.k,))

        for e in p_elem:
            x[e] = np.random.uniform(self.min_value, self.max_value) / 255

        return x