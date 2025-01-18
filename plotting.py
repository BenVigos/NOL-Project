import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.load("training_results.npz", allow_pickle=True)

# Convert NumPy arrays to Python lists
grid = data["hyperparameter_grid"].tolist()
max_train_acc_epochs = data["max_train_acc_epochs"].tolist()
max_train_accs = data["max_train_accs"].tolist()
final_train_accs = data["final_train_accs"].tolist()
min_train_loss_epochs = data["min_train_loss_epochs"].tolist()
min_train_losss = data["min_train_losss"].tolist()
final_train_losss = data["final_train_losss"].tolist()
max_test_acc_epochs = data["max_test_acc_epochs"].tolist()
max_test_accs = data["max_test_accs"].tolist()
final_test_accs = data["final_test_accs"].tolist()
min_test_loss_epochs = data["min_test_loss_epochs"].tolist()
min_test_losss = data["min_test_losss"].tolist()
final_test_losss = data["final_test_losss"].tolist()

# Helper function to plot
def plot_results(x, y, xlabel, ylabel, title, save_as=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', marker='x')
    plt.xscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if save_as:
        plt.savefig(save_as, dpi=300)
    plt.show()

# Plotting examples
# 1. Best training accuracies vs. hyperparameter combinations
plot_results(
    range(len(grid)),
    max_train_accs,
    xlabel="Hyperparameter Combination Index",
    ylabel="Max Train Accuracy",
    title="Max Training Accuracy for Each Hyperparameter Combination"
)

# 2. Final test accuracies vs. hyperparameter combinations
plot_results(
    range(len(grid)),
    final_test_accs,
    xlabel="Hyperparameter Combination Index",
    ylabel="Final Test Accuracy",
    title="Final Test Accuracy for Each Hyperparameter Combination"
)

# 3. Minimum training loss vs. hyperparameter combinations
plot_results(
    range(len(grid)),
    min_train_losss,
    xlabel="Hyperparameter Combination Index",
    ylabel="Min Train Loss",
    title="Minimum Training Loss for Each Hyperparameter Combination"
)

# 4. Minimum test loss vs. hyperparameter combinations
plot_results(
    range(len(grid)),
    min_test_losss,
    xlabel="Hyperparameter Combination Index",
    ylabel="Min Test Loss",
    title="Minimum Test Loss for Each Hyperparameter Combination"
)

print("Done. Plots generated successfully.")

# Extract learning rates from the hyperparameter grid
learning_rates = [combination[2] for combination in grid]  # Assuming learning rate is the third value in each combination

# Plot max training accuracy vs. learning rate
plot_results(
    learning_rates,
    max_train_accs,
    xlabel="Learning Rate",
    ylabel="Max Train Accuracy",
    title="Max Training Accuracy vs. Learning Rate"
)

# Plot final test accuracy vs. learning rate
plot_results(
    learning_rates,
    final_test_accs,
    xlabel="Learning Rate",
    ylabel="Final Test Accuracy",
    title="Final Test Accuracy vs. Learning Rate"
)

# Plot minimum training loss vs. learning rate
plot_results(
    learning_rates,
    min_train_losss,
    xlabel="Learning Rate",
    ylabel="Min Train Loss",
    title="Minimum Training Loss vs. Learning Rate"
)

# Plot minimum test loss vs. learning rate
plot_results(
    learning_rates,
    min_test_losss,
    xlabel="Learning Rate",
    ylabel="Min Test Loss",
    title="Minimum Test Loss vs. Learning Rate"
)

print("Plots of learning rates generated successfully.")

