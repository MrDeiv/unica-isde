from abc import ABC, abstractmethod
import numpy as np


class CDataPerturb(ABC):

    @abstractmethod
    def data_perturbation(self, x):
        raise NotImplementedError("Abstract method must be implemented before use " + self.__class__)

    def perturb_dataset(self, X):
        Xp = X.copy()
        """perturb = np.vectorize(self.data_perturbation)
        return perturb(X)"""

        for i in range(len(X)):
            Xp[i] = self.data_perturbation(Xp[i])

        return Xp


