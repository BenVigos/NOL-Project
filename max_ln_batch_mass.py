import numpy as np
import scipy.io

# Load data from the first .mat file
data_ada = scipy.io.loadmat('/Users/toprakkemalguneylioglu/Desktop/experiment/NOL-Project-Nesterov-concurrent/training_results_test_initial_ada.mat')

max_train_accs = data_ada['max_train_accs'].flatten()
max_test_acss = data_ada['max_test_accs'].flatten()
learning_rates = data_ada['learning_rates'].flatten()
batch_sizes = data_ada['batch_sizes'].flatten()

max_index = np.argmax(max_train_accs)

print("Max training learning rate: ", learning_rates[max_index])
print("Max training batch size: ", batch_sizes[max_index])
#print("Max training accs: ", np.max(max_train_accs))
#print('Max testing accs: ', np.max(max_test_acss))