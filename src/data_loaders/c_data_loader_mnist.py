from data_loaders import CDataLoader


class CDataLoaderMNIST(CDataLoader):
    """
    Loader for the MNIST handwritten digit data
    """

    def __init__(self, filename="../../data/mnist_train_small.csv"):
        self._filename = None
        self.filename = filename

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename):
        if not isinstance(filename, str):
            raise ValueError("Filename must be a valid string")

        self._filename = filename

    def load_data(self):
        pass
