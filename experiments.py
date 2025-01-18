import copy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # gives progression bars when running code

from activation_functions import logi, softmax
from data_loader import DataLoader
from loss_functions import mse_loss
from models import NeuralNetwork
from supplementary import Value, load_mnist
from training import train

# Set printing precision for NumPy so that we don't get needlessly many digits in our answers.
np.set_printoptions(precision=2)

# Get images and corresponding labels from the (fashion-)mnist dataset
data_dir = Path(__file__).resolve().parent / "data"
train_images, train_y = load_mnist(data_dir, kind='train')
test_images, test_y = load_mnist(data_dir, kind='t10k')

# Reshape each of the 60 000 images from a 28x28 image into a 784 vector.
# Rescale the values in the 784 to be in [0,1] instead of [0, 255].
train_images = train_images.reshape(60_000, 784) / 255
test_images = test_images.reshape(10_000, 784) / 255

# Labels are stored as numbers. For neural network training, we want one-hot encoding, i.e. the label should be a vector
# of 10 long with a one in the index corresponding to the digit.
train_labels = np.zeros((60_000, 10))
train_labels[np.arange(60_000), train_y] = 1
test_labels = np.zeros((10_000, 10))
test_labels[np.arange(10_000), test_y] = 1

# Take the first N images to train to speed up the training. This is useful when you want to test some new things.
# However, we need to work will the full dataset to show real results.
# training_subset = 100
# train_images = train_images[:training_subset]
# train_labels = train_labels[:training_subset]

# The data loader takes at every iteration batch_size items from the dataset. If it is not possible to take batch_size
# items, it takes whatever it still can. With 100 images in our dataset and a batch size of 32, it will be batches of
# 32, 32, 32, and 4.
train_dataset = list(zip(train_images, train_labels))
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=False)
train_dataset_size = len(train_dataset)

test_dataset = list(zip(test_images, test_labels))
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, drop_last=False)
test_dataset_size = len(test_dataset)


mass = 5e-1

epochs = 3

learning_rate = 3e-1

[neural_network, train_losses, test_losses, train_accuracies, test_accuracies] = train(train_loader, train_dataset_size, test_loader, test_dataset_size, mass=mass, epochs=epochs, learning_rate=learning_rate)


# Plot of train vs test losses on the same axes
plt.figure()
plt.title("Loss: train vs test")
plt.semilogy(np.array(range(1, epochs+1)), train_losses, label="train")
plt.semilogy(np.array(range(1, epochs+1)), test_losses, label="test")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plot of train vs test loss on the x-axis but with different y-axis
figure, ax1 = plt.subplots()
color = "tab:blue"
ax1.set_title("Loss: train vs test")
ax1.semilogy(np.array(range(1, epochs+1)), train_losses, color=color, label="train")
ax1.set_ylabel("Train loss", color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = "tab:orange"
ax2.semilogy(np.array(range(1, epochs+1)), test_losses, color=color, label="test")
ax2.set_ylabel("Test loss", color=color)
ax2.tick_params(axis='y', labelcolor=color)

figure.tight_layout()

# Plot of train vs test accuracies on the same axes
plt.figure()
plt.title("Accuracy: train vs test")
plt.plot(np.array(range(1, epochs+1)), train_accuracies, label="train")
plt.plot(np.array(range(1, epochs+1)), test_accuracies, label="test")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Plot of train vs test accuracies on the x-axis but with different y-axis
figure, ax1 = plt.subplots()
color = "tab:blue"
ax1.set_title("Accuracy: train vs test")
ax1.semilogy(np.array(range(1, epochs+1)), train_accuracies, color=color, label="train")
ax1.set_ylabel("Train loss", color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = "tab:orange"
ax2.semilogy(np.array(range(1, epochs+1)), test_accuracies, color=color, label="test")
ax2.set_ylabel("Test loss", color=color)
ax2.tick_params(axis='y', labelcolor=color)

figure.tight_layout()

# We take a random starting point for 10 subsequent images we want to take a greater look at.
r = np.random.randint(0, 9_990)

# We go over 10 images starting with r, plot them and show the prediction the network makes next to them.
plt.figure()
for i in range(9):
    plt.rcParams["figure.figsize"] = (15, 10)
    plt.subplot(3, 3, 1 + i)
    image = Value(np.array(test_images[r + i]), "x")
    plt.imshow(image.data.reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.text(-5, 45,
             f'True value:\n{test_labels[r + i]}: {test_y[r + i]}\n'
             f'Output:\n'
             f'[{neural_network(image)[0]:.2f} '  # needs __getitem__ method in Value class!
             f'{neural_network(image)[1]:.2f} '
             f'{neural_network(image)[2]:.2f} '
             f'{neural_network(image)[3]:.2f} '
             f'{neural_network(image)[4]:.2f}\n'
             f'{neural_network(image)[5]:.2f} '
             f'{neural_network(image)[6]:.2f} '
             f'{neural_network(image)[7]:.2f} '
             f'{neural_network(image)[8]:.2f} '
             f'{neural_network(image)[9]:.2f}]: {np.argmax(neural_network(image).data)}')

plt.subplots_adjust(hspace=.8)
plt.show()