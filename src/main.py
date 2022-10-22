import numpy as np
from matplotlib import pyplot as plt
from data_loaders import CDataLoaderMNIST
from data_perturb import CDataPerturbRandom
from classifiers import NMC

def plot_digit(image, shape=(28, 28)):
    plt.imshow(np.reshape(image, newshape=(28, 28)), 'gray')

def plot_ten_digits(x, y=None):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plot_digit(x[i, :])
        if y is not None:
            plt.title('Label: ' + str(y[i]))


dataset = CDataLoaderMNIST('../data/mnist_train_small.csv')
perturb = CDataPerturbRandom(k=784)
x, y = dataset.load_data()

nmc = NMC()
xtr, ytr, xts, yts = nmc.split_data(x, y, .6)

xp = perturb.perturb_dataset(x)

plot_ten_digits(xp)

plt.show()
