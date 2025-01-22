import copy

import numpy as np

from supplementary import Value


def weight_init_function(layer_size1: int, layer_size2: int):
    return np.random.uniform(-1, 1, (layer_size1, layer_size2))


def bias_init_function(layer_size: int):
    return np.random.uniform(-1, 1, layer_size)


def weight_delta_init_function(layer_size1: int, layer_size2: int):
    return np.zeros((layer_size1, layer_size2))


def bias_delta_init_function(layer_size: int):
    return np.zeros(layer_size)


class NeuralNetwork:
    r"""Neural network class.
    """

    def __init__(self, layers, activation_functions, seed=42, mass=0):
        self.number_of_layers = len(layers) - 1
        self.biases = []
        self.weights = []
        self.weight_deltas = []
        self.bias_deltas = []
        self.activation_functions = activation_functions
        self.mass = mass

        print(f"seed: {seed}")

        np.random.seed(seed)

        if len(activation_functions) != self.number_of_layers:
            raise ValueError("Number of activation functions should match the number of layers.")

        for i, size in enumerate(layers):
            if i > 0:
                self.biases.append(
                    Value(data=bias_init_function(layer_size=size), expr=f"$b^{{{i}}}$")
                )
                self.bias_deltas.append(
                    Value(data=bias_delta_init_function(layer_size=size), expr=f"$(\u03B4_b)^{{{i}}}$"))
            if i < self.number_of_layers:
                # We initialize the weights transposed
                self.weights.append(
                    Value(weight_init_function(layer_size1=size, layer_size2=layers[i + 1]), expr=f"$(W^T)^{{{i}}}$")
                )
                self.weight_deltas.append(
                    Value(weight_delta_init_function(layer_size1=size, layer_size2=layers[i + 1]),
                          expr=f"$((\u03B4^T)_W)^{{{i}}}$"))

    def __call__(self, x):
        for (weight, weight_delta, bias, bias_delta, activation_function) in zip(self.weights, self.weight_deltas, self.biases, self.bias_deltas, self.activation_functions):
            x = activation_function(x @ (weight + (- self.mass) * weight_delta) + bias + (- self.mass) * bias_delta)
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

    def nesterov_descent(self, learning_rate):
        for i, weight in enumerate(self.weights):
            delta = self.weight_deltas[i]
            delta.data = delta.data * self.mass + learning_rate * weight.grad
            weight.data -= delta.data
        for i, bias in enumerate(self.biases):
            delta = self.bias_deltas[i]
            delta.data = delta.data * self.mass + learning_rate * bias.grad
            bias.data -= bias.grad

    def adagrad(self, learning_rate, cw, cb):
        for index, weight in enumerate(self.weights):
            cw[index] += np.square(weight.grad)
            weight.data -= (learning_rate / np.sqrt(1e-7 + cw[index])) * weight.grad
        for k, bias in enumerate(self.biases):
            cb[k] += np.square(bias.grad)
            bias.data -= (learning_rate / np.sqrt(1e-7 + cb[k])) * bias.grad

