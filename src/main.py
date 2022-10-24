import numpy as np
from matplotlib import pyplot as plt
from data_loaders import CDataLoaderMNIST
from data_perturb import CDataPerturbRandom, CDataPerturbGaussian
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


# Load data
dataset = CDataLoaderMNIST('../data/mnist_train_small.csv')
x, y = dataset.load_data()

# Classifier
nmc = NMC()
xtr, ytr, xts, yts = nmc.split_data(x, y, .6)

centroids = nmc.fit(xtr, ytr)

y_pred = nmc.predict(xts)
acc = nmc.accuracy(y_pred, yts)
print(acc)

# Perturbations
# Random noise
k_vals = [0, 10, 20, 50, 100, 200, 500]
accur_1 = []
for k in k_vals:
    pert = CDataPerturbRandom(k=k)
    y_pred = nmc.predict(pert.perturb_dataset(xts))
    accur_1.append(nmc.accuracy(y_pred, yts))

# Gaussian noise
sigma_vals = [10, 20, 200, 200, 500]
accur_2 = []
for s in sigma_vals:
    pert = CDataPerturbGaussian(sigma=s)
    y_pred = nmc.predict(pert.perturb_dataset(xts))
    accur_2.append(nmc.accuracy(y_pred, yts))

# Plot
plt.figure()

# Leftmost plot
plt.subplot(1,2,1)
plt.ylabel("K")
plt.xlabel("Accuracy")
plt.plot(accur_1, k_vals)

# Rightmost plot
plt.subplot(1,2,2)
plt.ylabel("Sigma")
plt.xlabel("Accuracy")
plt.plot(accur_2, sigma_vals)

plt.show()
