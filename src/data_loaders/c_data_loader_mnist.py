from data_loaders import CDataLoader
import pandas as pd
import numpy as np


class CDataLoaderMNIST(CDataLoader):
    """
    Loader for the MNIST handwritten digit data
    """

    def __init__(self, filename="../../data/mnist_train_small.csv"):
        self._filename = None
        self.filename = filename
        self._width = 28
        self._height = 28

    @property
    def filename(self):
        return self._filename

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @filename.setter
    def filename(self, filename):
        if not isinstance(filename, str):
            raise ValueError("Filename must be a valid string")

        self._filename = filename

    def load_data(self):
        data = pd.read_csv(self.filename)
        data = np.array(data)

        y = data[:, 0]  # all rows and first column
        x = data[:, 1:] / 255  # all rows and remaing columns

        return x, y
