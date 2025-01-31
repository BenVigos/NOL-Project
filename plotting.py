import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Load data from the first .mat file
data1 = scipy.io.loadmat('/Users/toprakkemalguneylioglu/Desktop/experiment/nesterov_results/training_results_seeds_nest_100.mat')
seeds1 = data1['seeds'].flatten()
learning_rates1 = data1['learning_rates'].flatten()
max_test_accs1 = data1['max_test_accs'].flatten()
max_train_accs1 = data1['max_train_accs'].flatten()
final_train_loss_1 = data1['final_train_losss'].flatten()
final_test_loss_1 = data1['final_test_losss'].flatten()

# Load data from the second .mat file
data2 = scipy.io.loadmat('/Users/toprakkemalguneylioglu/Desktop/experiment/ada_results/training_results_seeds_ada_100.mat')
seeds2 = data2['seeds'].flatten()
learning_rates2 = data2['learning_rates'].flatten()
max_test_accs2 = data2['max_test_accs'].flatten()
max_train_accs2 = data2['max_train_accs'].flatten()
final_train_loss_2 = data2['final_train_losss'].flatten()
final_test_loss_2 = data2['final_test_losss'].flatten()

# Create a figure with 2 rows and 2 columns of subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Nesterov vs Adagrad: dependence on initial parameters", fontsize=16)

# Plot for Data 1
unique_seeds1 = np.unique(seeds1)
for seed in unique_seeds1:
    seed_indices = seeds1 == seed
    seed_learning_rates = learning_rates1[seed_indices]
    seed_max_test_accs = max_test_accs1[seed_indices]
    seed_max_train_accs = max_train_accs1[seed_indices]
    seed_final_test_loss = final_test_loss_1[seed_indices]
    seed_final_train_loss = final_train_loss_1[seed_indices]
    sort_indices = np.argsort(seed_learning_rates)
    sorted_learning_rates = seed_learning_rates[sort_indices]
    sorted_max_test_accs = seed_max_test_accs[sort_indices]
    sorted_max_train_accs = seed_max_train_accs[sort_indices]
    sorted_final_test_loss = seed_final_test_loss[sort_indices]
    sorted_final_train_loss = seed_final_train_loss[sort_indices]

    axs[0, 0].plot(sorted_learning_rates, sorted_max_test_accs, '-', label=f'Seed {int(seed)}')
    axs[0, 1].plot(sorted_learning_rates, sorted_max_train_accs, '-', label=f'Seed {int(seed)}')

# Configure subplots for Data 1
axs[0, 0].set_title('Nesterov: Maximum Test Accuracy')
axs[0, 0].set_xscale('log')
axs[0, 0].set_xlabel('Learning Rate')
axs[0, 0].set_ylabel('Maximum Accuracy')
axs[0, 0].grid(True)

axs[0, 1].set_title('Nesterov: Maximum Train Accuracy')
axs[0, 1].set_xscale('log')
axs[0, 1].set_xlabel('Learning Rate')
axs[0, 1].grid(True)

# Plot for Data 2
unique_seeds2 = np.unique(seeds2)
for seed in unique_seeds2:
    seed_indices = seeds2 == seed
    seed_learning_rates = learning_rates2[seed_indices]
    seed_max_test_accs = max_test_accs2[seed_indices]
    seed_max_train_accs = max_train_accs2[seed_indices]
    seed_final_test_loss = final_test_loss_2[seed_indices]
    seed_final_train_loss = final_train_loss_2[seed_indices]
    sort_indices = np.argsort(seed_learning_rates)
    sorted_learning_rates = seed_learning_rates[sort_indices]
    sorted_max_test_accs = seed_max_test_accs[sort_indices]
    sorted_max_train_accs = seed_max_train_accs[sort_indices]
    sorted_final_test_loss = seed_final_test_loss[sort_indices]
    sorted_final_train_loss = seed_final_train_loss[sort_indices]

    axs[1, 0].plot(sorted_learning_rates, sorted_max_test_accs, '-', label=f'Seed {int(seed)}')
    axs[1, 1].plot(sorted_learning_rates, sorted_max_train_accs, '-', label=f'Seed {int(seed)}')

# Configure subplots for Data 2
axs[1, 0].set_title('Adagrad: Maximum Test Accuracy')
axs[1, 0].set_xscale('log')
axs[1, 0].set_xlabel('Learning Rate')
axs[1, 0].set_ylabel('Maximum Accuracy')
axs[1, 0].grid(True)

axs[1, 1].set_title('Adagrad: Maximum Train Accuracy')
axs[1, 1].set_xscale('log')
axs[1, 1].set_xlabel('Learning Rate')
axs[1, 1].grid(True)

# Add legends
for ax in axs.flat:
    ax.legend()

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show the plots
plt.show()
