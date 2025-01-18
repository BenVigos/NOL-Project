import numpy as np

data = np.load("training_results.npz", allow_pickle=True)

grid = data["hyperparameter_grid"]
print("Done")