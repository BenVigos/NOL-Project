import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

# Load the data
data = np.load("/Users/toprakkemalguneylioglu/Desktop/experiment/nesterov_results/training_results_seeds_nest.npz", allow_pickle=True)

# Extract the data
batches = data["batch_sizes"]
learning_rates = data["learning_rates"]
epochs = np.ones(len(data)-1)*100
min_train_losss = data["min_train_losss"]
min_train_loss_epochs = data["min_train_loss_epochs"]
final_train_losss = data["final_train_losss"]
min_test_losss = data["min_test_losss"]
final_test_losss = data["final_test_losss"]
min_test_loss_epochs = data["min_test_loss_epochs"]


max_train_accs = data["max_train_accs"]
max_train_acc_epochs = data["max_train_acc_epochs"]
final_train_accs = data["final_train_accs"]
max_test_acc_epochs = data["max_test_accs"]
max_test_accs = data["max_test_acc_epochs"]
final_test_accs = data["final_test_accs"]

# Filter learning rates and masses where min_test_losss > 100
exceeding_threshold_indices = np.where(min_test_losss > 100)[0]
learning_rates_exceeding = np.array(learning_rates)[exceeding_threshold_indices]
batches_exceeding = np.array(batches)[exceeding_threshold_indices]

# Print the results
for lr, batch, loss in zip(learning_rates_exceeding, batches_exceeding, min_test_losss[exceeding_threshold_indices]):
    print(f"Learning Rate: {lr}, Batches: {batch}, Min Test Loss: {loss}")

print("Done")


# Save to a .mat file
savemat("training_results_test.mat", data)


# Helper function for 3D plotting
def plot_3d(x, y, z, xlabel, ylabel, zlabel, title, save_as=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Take the logarithm of x for plotting
    log_x = np.log10(x)

    # Create scatter plot
    scatter = ax.scatter(log_x, y, z, c=z, cmap='viridis', s=50)
    fig.colorbar(scatter, ax=ax, label=zlabel)

    # Set axis labels and ticks
    ax.set_xlabel(f"Log10({xlabel})")
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    # Custom ticks for x-axis to reflect actual values
    ax.set_xticks(np.log10(x))  # Set ticks at log-transformed positions
    ax.set_xticklabels([f"{val:.1e}" for val in x])  # Show original values

    # Save the plot if required
    if save_as:
        plt.savefig(save_as, dpi=300)
    plt.show()


def plot_3d_mesh(x, y, z, xlabel, ylabel, zlabel, title, save_as=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Take the logarithm of x for plotting
    log_x = np.log10(x)

    # Create unique sorted values for x and y
    unique_log_x = np.sort(np.unique(log_x))
    unique_y = np.sort(np.unique(y))

    # Create a grid for the surface plot
    X, Y = np.meshgrid(unique_log_x, unique_y)
    Z = np.full_like(X, np.nan, dtype=np.float64)  # Initialize Z with NaNs

    # Populate Z values based on the original data
    for i, (lx, mass) in enumerate(zip(log_x, y)):
        xi = np.where(unique_log_x == lx)[0][0]
        yi = np.where(unique_y == mass)[0][0]
        Z[yi, xi] = z[i]  # Assign the correct z value

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)
    fig.colorbar(surf, ax=ax, label=zlabel)

    # Set axis labels and ticks
    ax.set_xlabel(f"Log10({xlabel})")
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    # Custom ticks for x-axis to reflect actual values
    ax.set_xticks(unique_log_x)
    ax.set_xticklabels([f"{10**val:.1e}" for val in unique_log_x])

    # Save the plot if required
    if save_as:
        plt.savefig(save_as, dpi=300)
    plt.show()





# Plot minimum training loss
plot_3d_mesh(
    learning_rates, batches, min_train_losss,
    xlabel="Learning Rate (log scale)",
    ylabel="Batch Size",
    zlabel="Min Train Loss",
    title="Minimum Training Loss vs Learning Rate and Batch Size"
)

# Plot final training loss
plot_3d_mesh(
    learning_rates, batches, final_train_losss,
    xlabel="Learning Rate (log scale)",
    ylabel="Batch Size",
    zlabel="Final Train Loss",
    title="Final Training Loss vs Learning Rate and Batch Size"
)

# Plot best test loss
plot_3d_mesh(
    learning_rates, batches, min_test_losss,
    xlabel="Learning Rate (log scale)",
    ylabel="Batch Size",
    zlabel="Min Test Loss",
    title="Minimum Test Loss vs Learning Rate and Batch Size"
)

# Plot final test loss
plot_3d_mesh(
    learning_rates, batches, final_test_losss,
    xlabel="Learning Rate (log scale)",
    ylabel="Batch Size",
    zlabel="Final Test Loss",
    title="Final Test Loss vs Learning Rate and Batch Size"
)

# Plot best training accuracy
plot_3d_mesh(
    learning_rates, batches, max_train_accs,
    xlabel="Learning Rate (log scale)",
    ylabel="Batch Size",
    zlabel="Maximum Training accuracy",
    title="Maximum Training accuracy vs Learning Rate and Batch Size"
)

plot_3d_mesh(
    learning_rates, batches, final_train_accs,
    xlabel="Learning Rate (log scale)",
    ylabel="Batch Size",
    zlabel="Final Training accuracy",
    title="Final Training accuracy vs Learning Rate and Batch Size"
)

plot_3d_mesh(
    learning_rates, batches, max_test_accs,
    xlabel="Learning Rate (log scale)",
    ylabel="Batch Size",
    zlabel="Maximum Test accuracy",
    title="Maximum Test accuracy vs Learning Rate and Batch Size"
)

plot_3d_mesh(
    learning_rates, batches, final_test_accs,
    xlabel="Learning Rate (log scale)",
    ylabel="Batch Size",
    zlabel="Final Test accuracy",
    title="Final Test accuracy vs Learning Rate and Batch Size"
)
