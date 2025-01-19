import copy
import itertools
from pathlib import Path
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

from activation_functions import logi, softmax
from data_loader import DataLoader
from loss_functions import mse_loss
from models import NeuralNetwork
from supplementary import Value, load_mnist
from training import train

np.set_printoptions(precision=2)

data_dir = Path(__file__).resolve().parent / "data"
train_images, train_y = load_mnist(data_dir, kind='train')
test_images, test_y = load_mnist(data_dir, kind='t10k')

train_images = train_images.reshape(60_000, 784) / 255
test_images = test_images.reshape(10_000, 784) / 255

train_labels = np.zeros((60_000, 10))
train_labels[np.arange(60_000), train_y] = 1
test_labels = np.zeros((10_000, 10))
test_labels[np.arange(10_000), test_y] = 1

train_dataset = list(zip(train_images, train_labels))

test_dataset = list(zip(test_images, test_labels))

masses = np.linspace(0, 1, 10)
batch_sizes = [128]
learning_rates = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 10]
epochs = 2
hyperparameter_grid = list(itertools.product(masses, batch_sizes, learning_rates))


def simulate_and_record(lock, shared_results, mass, batch_size, learning_rate, i, epochs):
    """Run a single training simulation and record the results."""
    # Create a new DataLoader instance for each process
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    train_dataset_size = len(train_dataset)
    test_dataset_size = len(test_dataset)

    [neural_network, train_losses, test_losses, train_accuracies, test_accuracies] = train(
        train_loader, train_dataset_size, test_loader, test_dataset_size,
        mass=mass, epochs=epochs, learning_rate=learning_rate, i=i
    )

    with lock:
        shared_results['masses'].append(mass)
        shared_results['learning_rates'].append(learning_rate)
        shared_results['batch_sizes'].append(batch_size)

        shared_results['max_train_acc_epochs'].append(train_accuracies.index(max(train_accuracies)) + 1)
        shared_results['max_train_accs'].append(max(train_accuracies))
        shared_results['final_train_accs'].append(train_accuracies[-1])
        shared_results['min_train_loss_epochs'].append(train_losses.index(min(train_losses)) + 1)
        shared_results['min_train_losss'].append(min(train_losses))
        shared_results['final_train_losss'].append(train_losses[-1])

        shared_results['max_test_acc_epochs'].append(test_accuracies.index(max(test_accuracies)) + 1)
        shared_results['max_test_accs'].append(max(test_accuracies))
        shared_results['final_test_accs'].append(test_accuracies[-1])
        shared_results['min_test_loss_epochs'].append(test_losses.index(min(test_losses)) + 1)
        shared_results['min_test_losss'].append(min(test_losses))
        shared_results['final_test_losss'].append(test_losses[-1])

    print(f"{i}: Done with (mass | learning rate | epochs) : ({mass} | {learning_rate} | {batch_size}) \n "
          f"min test loss: {min(test_losses)}, final test loss: {test_losses[-1]}")


def run_batch(batch, lock, shared_results, epochs):
    """Run a batch of simulations."""
    processes = []
    for i, (mass, batch_size, learning_rate) in enumerate(batch):
        print(f"{i}: Started with (mass | learning rate | epochs) : ({mass} | {learning_rate} | {batch_size})")
        p = mp.Process(target=simulate_and_record, args=(lock, shared_results, mass, batch_size, learning_rate, i, epochs))
        processes.append(p)
        p.start()

    # Wait for all processes in the batch to complete
    for p in processes:
        p.join()


def main():
    # Create shared structures and lock
    manager = mp.Manager()
    shared_results = manager.dict({
        'masses': manager.list(),
        'learning_rates': manager.list(),
        'batch_sizes': manager.list(),
        'max_train_acc_epochs': manager.list(),
        'max_train_accs': manager.list(),
        'final_train_accs': manager.list(),
        'min_train_loss_epochs': manager.list(),
        'min_train_losss': manager.list(),
        'final_train_losss': manager.list(),
        'max_test_acc_epochs': manager.list(),
        'max_test_accs': manager.list(),
        'final_test_accs': manager.list(),
        'min_test_loss_epochs': manager.list(),
        'min_test_losss': manager.list(),
        'final_test_losss': manager.list(),
    })
    lock = manager.Lock()

    # Specify the batch size (adjust based on your system's performance)
    sim_batch_size = 10

    # Split hyperparameter_grid into batches
    batches = [hyperparameter_grid[i:i + sim_batch_size] for i in range(0, len(hyperparameter_grid), sim_batch_size)]

    # Process each batch
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}")
        run_batch(batch, lock, shared_results, epochs=epochs)

    # Save results
    np.savez(
        "training_results_masses_short.npz",

    masses = shared_results["masses"],
    learning_rates = shared_results["learning_rates"],
    batch_sizes = shared_results["batch_sizes"],
    max_train_acc_epochs = shared_results["max_train_acc_epochs"],
    max_train_accs = shared_results["max_train_accs"],
    final_train_accs = shared_results["final_train_accs"],
    min_train_loss_epochs = shared_results["min_train_loss_epochs"],
    min_train_losss = shared_results["min_train_losss"],
    final_train_losss = shared_results["final_train_losss"],
    max_test_acc_epochs = shared_results["max_test_acc_epochs"],
    max_test_accs = shared_results["max_test_accs"],
    final_test_accs = shared_results["final_test_accs"],
    min_test_loss_epochs = shared_results["min_test_loss_epochs"],
    min_test_losss = shared_results["min_test_losss"],
    final_test_losss = shared_results["final_test_losss"],
    )


if __name__ == "__main__":
    main()
