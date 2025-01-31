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


def train(train_loader, train_dataset_size, test_loader, test_dataset_size, mass=0, learning_rate=3e-3, epochs=10, i=-1, seed=42):
    neural_network = NeuralNetwork(
        layers=[784, 256, 128, 64, 10],
        activation_functions=[logi, logi, logi, softmax], mass=mass, seed=seed
    )
    # Set training configuration
    learning_rate = learning_rate
    epochs = epochs
    # Do the full training algorithm
    cw = [0 for weight in neural_network.weights]
    cb = [0 for bias in neural_network.biases]
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        # (Re)set the training loss for this epoch.
        train_loss = 0.0
        correctly_classified = 0
        for batch in train_loader:
            # Reset the gradients so that we start fresh.
            neural_network.reset_gradients()

            # Get the images and labels from the batch
            images = np.vstack([image for (image, _) in batch])
            labels = np.vstack([label for (_, label) in batch])

            # Wrap images and labels in a Value class.
            images = Value(images, expr="X")
            labels = Value(labels, expr="Y")

            # Compute what the model says is the label.
            output = neural_network(images)

            # Compute the loss for this batch.
            loss = mse_loss(
                output,
                labels
            )

            # Do backpropagation
            loss.backward()

            # Update the weights and biases using the chosen algorithm, in this case gradient descent.
            neural_network.adagrad(learning_rate, cw, cb)

            # Store the loss for this batch.
            train_loss += loss.data

            # Store accuracies for extra interpretability
            true_classification = np.argmax(
                labels.data,
                axis=1
            )
            predicted_classification = np.argmax(
                output.data,
                axis=1
            )
            correctly_classified += np.sum(true_classification == predicted_classification)

        # Store the loss and average accuracy for the entire epoch.
        train_losses.append(train_loss)
        train_accuracies.append(correctly_classified / train_dataset_size)

        test_loss = 0.0
        correctly_classified = 0
        for batch in tqdm(test_loader, desc=f"Testing epoch {epoch}"):
            # Get the images and labels from the batch
            images = np.vstack([image for (image, _) in batch])
            labels = np.vstack([label for (_, label) in batch])

            # Wrap images and labels in a Value class.
            images = Value(images, expr="X")
            labels = Value(labels, expr="Y")

            # Compute what the model says is the label.
            output = neural_network(images)

            # Compute the loss for this batch.
            loss = mse_loss(
                output,
                labels
            )

            # Store the loss for this batch.
            test_loss += loss.data

            # Store accuracies for extra interpretability
            true_classification = np.argmax(
                labels.data,
                axis=1
            )
            predicted_classification = np.argmax(
                output.data,
                axis=1
            )
            correctly_classified += np.sum(true_classification == predicted_classification)

        test_losses.append(test_loss)
        test_accuracies.append(correctly_classified / test_dataset_size)

    return [neural_network, train_losses, test_losses, train_accuracies, test_accuracies]
