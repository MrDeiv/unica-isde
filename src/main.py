import numpy as np
from matplotlib import pyplot as plt
from data_loaders import CDataLoaderMNIST
from data_perturb import CDataPerturbRandom, CDataPerturbGaussian
from classifiers import NMC


def plot_digit(image, shape=(28, 28)):
    plt.imshow(np.reshape(image, newshape=shape), 'gray')


def plot_ten_digits(x, y=None):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plot_digit(x[i, :])
        if y is not None:
            plt.title('Label: ' + str(y[i]))


# Load data
dataset = CDataLoaderMNIST('../data/mnist_train_small.csv')
x, y = dataset.load_data()

# Classifier
nmc = NMC()
xtr, ytr, xts, yts = nmc.split_data(x, y, .6)

centroids = nmc.fit(xtr, ytr)

y_pred = nmc.predict(xts)
acc = nmc.accuracy(y_pred, yts)
print("Initial accuracy:", acc*100, "%")

# Perturbations
# Random noise
k_vals = np.array([0, 10, 20, 50, 100, 200, 500])
accur_1 = np.zeros(k_vals.size)
pert = CDataPerturbRandom()
for i, k in enumerate(k_vals):
    pert.k = k
    y_pred = nmc.predict(pert.perturb_dataset(xts))
    accur_1[i] = nmc.accuracy(y_pred, yts)

# Gaussian noise
pert = CDataPerturbGaussian(sigma=0.5)
sigma_vals = np.array([10, 20, 200, 200, 500])
accur_2 = np.zeros(sigma_vals.size)
for i, s in enumerate(sigma_vals):
    pert.sigma = sigma_vals[i]
    y_pred = nmc.predict(pert.perturb_dataset(xts))
    accur_2[i] = nmc.accuracy(y_pred, yts)

# Plot
plt.figure()

# Leftmost plot
plt.subplot(1,2,1)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.plot(k_vals, accur_1)

# Rightmost plot
plt.subplot(1,2,2)
plt.xlabel("Sigma")
plt.ylabel("Accuracy")
plt.plot(sigma_vals, accur_2)

plt.show()
