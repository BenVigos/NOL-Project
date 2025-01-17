import numpy as np

from supplementary import Value


def weight_init_function(layer_size1: int, layer_size2: int):
    return np.random.uniform(-1, 1, (layer_size1, layer_size2))


def bias_init_function(layer_size: int):
    return np.random.uniform(-1, 1, layer_size)


class NeuralNetwork:
    r"""Neural network class.
    """
    def __init__(self, layers, activation_functions):
        self.number_of_layers = len(layers) - 1
        self.biases = []
        self.weights = []
        self.activation_functions = activation_functions

        if len(activation_functions) != self.number_of_layers:
            raise ValueError("Number of activation functions should match the number of layers.")

        for i, size in enumerate(layers):
            if i > 0:
                self.biases.append(
                    Value(data=bias_init_function(layer_size=size), expr=f"$b^{{{i}}}$")
                )
            if i < self.number_of_layers:
                # We initialize the weights transposed
                self.weights.append(
                    Value(weight_init_function(layer_size1=size, layer_size2=layers[i+1]), expr=f"$(W^T)^{{{i}}}$")
                )

    def __call__(self, x):
        for (weight, bias, activation_function) in zip(self.weights, self.biases, self.activation_functions):
            x = activation_function(x @ weight + bias)
        return x

    def reset_gradients(self):
        for weight in self.weights:
            weight.reset_grad()
        for bias in self.biases:
            bias.reset_grad()

    def gradient_descent(self, learning_rate):
        for weight in self.weights:
            weight.data -= learning_rate * weight.grad
        for bias in self.biases:
            bias.data -= learning_rate * bias.grad

    def adagrad(self, learning_rate, cw, cb):
        for weight, c in zip(self.weights, cw):
            c += np.square(weight.grad)
            weight.data -= (learning_rate / np.sqrt(0.1 + c - np.square(weight.grad))) * weight.grad
        for bias, c2 in zip(self.weights, cb):
            c2 += np.square(bias.grad)
            bias.data -= (learning_rate / np.sqrt(0.1 + c2 - np.square(bias.grad))) * bias.grad
        return cw, cb
