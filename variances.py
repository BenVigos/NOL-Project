import numpy as np
import matplotlib.pyplot as plt
from models import NeuralNetwork
from activation_functions import logi, softmax

# Load the saved neural networks and learning rates
data = np.load("/Users/toprakkemalguneylioglu/Desktop/experiment/nesterov_results/sim_nn_nest.npz", allow_pickle=True)
saved_nn = data["sim_nn"].item()  # Extract the saved neural networks dictionary
results = np.load("/Users/toprakkemalguneylioglu/Desktop/experiment/nesterov_results/results_summary_nest.npz", allow_pickle=True)
max_test_acc_learning_rate = results["max_test_acc_learning_rate"].item()
max_train_acc_learning_rate = results["max_train_acc_learning_rate"].item()

# Initialize variables for means and variances
mean_weights = [0 for _ in range(4)]
mean_biases = [0 for _ in range(4)]
sample_var_weights_matrices = [0 for _ in range(4)]
sample_var_biases_matrices = [0 for _ in range(4)]

def freedman_diaconis_bins(data):
    n = len(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)  # Interquartile range
    bin_width = 2 * iqr * n ** (-1 / 3)
    if bin_width == 0:
        return 1
    num_bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return max(num_bins, 1)


filtered_nn = {key: value for key, value in saved_nn.items() if key[1] == max_test_acc_learning_rate}
n = len(filtered_nn)

# Compute means and variances for weights and biases
for layer in range(4):
    for _, value in filtered_nn.items():
        mean_weights[layer] += value.weights[layer].data
        mean_biases[layer] += value.biases[layer].data

    mean_weights[layer] /= n
    mean_biases[layer] /= n

    '''for _, value in filtered_nn.items():
        sample_var_weights[layer] += (np.linalg.norm(value.weights[layer].data - mean_weights[layer]) ** 2) / n
        sample_var_biases[layer] += (np.linalg.norm(value.biases[layer].data - mean_biases[layer]) ** 2) / n'''

    var_matrix = 0 * filtered_nn[(914, max_test_acc_learning_rate)].weights[layer].data
    var_bias = 0 * filtered_nn[(914, max_test_acc_learning_rate)].biases[layer].data
    for _, value in filtered_nn.items():
        var_matrix += np.square(value.weights[layer].data - mean_weights[layer]) / (n - 1)
        var_bias += np.square(value.biases[layer].data - mean_biases[layer]) / (n - 1)

    sample_var_weights_matrices[layer] = var_matrix
    sample_var_biases_matrices[layer] = var_bias


seed_nn = {}

for seed in [914, 693, 640, 556, 78, 431, 199, 130, 81, 43]:
    nn = NeuralNetwork(
        layers=[784, 256, 128, 64, 10],
        activation_functions=[logi, logi, logi, softmax], mass=0, seed=seed)
    seed_nn[seed] = nn

n2 = len(seed_nn)
mean_weights_seed = [0 for _ in range(4)]
mean_biases_seed = [0 for _ in range(4)]
sample_var_weights_matrices_seed = [0 for _ in range(4)]
sample_var_biases_matrices_seed = [0 for _ in range(4)]

# Compute means and variances for weights and biases
for layer in range(4):
    for _, value in seed_nn.items():
        mean_weights_seed[layer] += value.weights[layer].data
        mean_biases_seed[layer] += value.biases[layer].data

    mean_weights_seed[layer] /= n2
    mean_biases_seed[layer] /= n2

    var_matrix = 0 * seed_nn[914].weights[layer].data
    var_bias = 0 * seed_nn[914].biases[layer].data

    for _, value in seed_nn.items():
        var_matrix += np.square(value.weights[layer].data - mean_weights[layer]) / (n2 - 1)
        var_bias += np.square(value.biases[layer].data - mean_biases[layer]) / (n2 - 1)

    sample_var_weights_matrices_seed[layer] = var_matrix
    sample_var_biases_matrices_seed[layer] = var_bias



fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Flatten axes array for easier indexing
axes = axes.flatten()


for layer in range(4):
    flatten_data_filtered = sample_var_biases_matrices[layer].flatten()
    flatten_data_seed = sample_var_biases_matrices_seed[layer].flatten()
    num_bins_filtered = freedman_diaconis_bins(flatten_data_filtered)
    num_bins_seed = freedman_diaconis_bins(flatten_data_seed)

    # Plot histogram for filtered_nn
    axes[layer].hist(
        flatten_data_filtered,
        bins=num_bins_filtered,
        color='skyblue',
        edgecolor='black',
        alpha=0.5,
        label='Trained NN'
    )

    # Plot histogram for seed_nn
    axes[layer].hist(
        flatten_data_seed,
        bins=num_bins_seed,
        color='orange',
        edgecolor='black',
        alpha=0.5,
        label='Initial NN'
    )

    axes[layer].grid(axis='y', linestyle='--', alpha=0.7)
    axes[layer].set_title(f'Layer {layer} - Variance Distribution of Biases for Nesterov', fontsize=12)
    axes[layer].set_xlabel('Variance', fontsize=10)
    axes[layer].set_ylabel('Frequency', fontsize=10)
    axes[layer].legend(loc='upper right', fontsize=9)

    '''# Set x-axis limit for layer 0
    if layer == 0:  # Layer 0 is indexed as 0
        axes[layer].set_xlim(0, 1.2)'''

plt.tight_layout()
plt.show()
