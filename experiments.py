import copy
import itertools
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
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)
train_dataset_size = len(train_dataset)

test_dataset = list(zip(test_images, test_labels))
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, drop_last=False)
test_dataset_size = len(test_dataset)

#An array of the epochs at which the best train accuracies were achieved
max_train_acc_epochs = []
#An array of the best train accuracies
max_train_accs = []
#An array of the final train accuracies (last epoch)
final_train_accs = []
#An array of the epochs at which the best train losses were achieved
min_train_loss_epochs = []
#An array of the best train losses
min_train_losss = []
#An array of the final train losses (last epoch)
final_train_losss = []

#An array of the epochs at which the best test accuracies were achieved
max_test_acc_epochs = []
#An array of the best test accuracies
max_test_accs = []
#An array of the final test losses (last epoch)
final_test_accs = []
#An array of the epochs at which the best test losses were achieved
min_test_loss_epochs = []
#An array of the best test losses
min_test_losss = []
#An array of the final test losses (last epoch)
final_test_losss = []

# DONT CHANGE ANYTHING ABOVE THIS LINE

masses = [0, 3e-1, 4e-1, 5e-1, 7e-1, 9e-1]
epochss = [1]
learning_rates = [3e-3, 3e-1, 3e-2, 5e-1]

hyperparameter_grid = list(itertools.product(masses, epochss, learning_rates))

i = 0
for mass, epochs, learning_rate in hyperparameter_grid:
    print(f"{i}: Training with (mass | learning rate | epochs) : ({mass} | {learning_rate} | {epochs})")
    i+= 1
    [neural_network, train_losses, test_losses, train_accuracies, test_accuracies] = train(train_loader,
                                                                                           train_dataset_size,
                                                                                           test_loader,
                                                                                           test_dataset_size,
                                                                                           mass=mass, epochs=epochs,
                                                                                           learning_rate=learning_rate)

    max_train_acc_epochs.append(train_accuracies.index(max(train_accuracies)) + 1)
    max_train_accs.append(train_accuracies[max_train_acc_epochs[-1] - 1])
    final_train_accs.append(train_accuracies[-1])
    min_train_loss_epochs.append(train_losses.index(min(train_losses)) + 1)
    min_train_losss.append(train_losses[min_train_loss_epochs[-1] - 1])
    final_train_losss.append(train_losses[-1])

    max_test_acc_epochs.append(test_accuracies.index(max(test_accuracies)) + 1)
    max_test_accs.append(test_accuracies[max_test_acc_epochs[-1] - 1])
    final_test_accs.append(test_accuracies[-1])
    min_test_loss_epochs.append(test_losses.index(min(test_losses)) + 1)
    min_test_losss.append(test_losses[min_test_loss_epochs[-1] - 1])
    final_test_losss.append(test_losses[-1])


#Save the data

max_train_acc_epochs = np.array(max_train_acc_epochs)
max_train_accs = np.array(max_train_accs)
final_train_accs = np.array(final_train_accs)
min_train_loss_epochs = np.array(min_train_loss_epochs)
min_train_losss = np.array(min_train_losss)
final_train_losss = np.array(final_train_losss)

max_test_acc_epochs = np.array(max_test_acc_epochs)
max_test_accs = np.array(max_test_accs)
final_test_accs = np.array(final_test_accs)
min_test_loss_epochs = np.array(min_test_loss_epochs)
min_test_losss = np.array(min_test_losss)
final_test_losss = np.array(final_test_losss)

# Convert hyperparameter grid to a numpy array
hyperparameter_grid = np.array(hyperparameter_grid, dtype=object)

np.savez(
    "training_results.npz",
    max_train_acc_epochs=max_train_acc_epochs,
    max_train_accs=max_train_accs,
    final_train_accs=final_train_accs,
    min_train_loss_epochs=min_train_loss_epochs,
    min_train_losss=min_train_losss,
    final_train_losss=final_train_losss,
    max_test_acc_epochs=max_test_acc_epochs,
    max_test_accs=max_test_accs,
    final_test_accs=final_test_accs,
    min_test_loss_epochs=min_test_loss_epochs,
    min_test_losss=min_test_losss,
    final_test_losss=final_test_losss,
    hyperparameter_grid=hyperparameter_grid
)

