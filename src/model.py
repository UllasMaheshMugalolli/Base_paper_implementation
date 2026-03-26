"""Lightweight feed-forward model implemented with pure Python / NumPy."""

import math
import random


def _dot(a, b):
    """Dot product of two equal-length lists."""
    return sum(x * y for x, y in zip(a, b))


def _sigmoid(z):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + math.exp(-z))


def _relu(z):
    """ReLU activation function."""
    return max(0.0, z)


def _relu_deriv(z):
    return 1.0 if z > 0 else 0.0


def _sigmoid_deriv(z):
    s = _sigmoid(z)
    return s * (1.0 - s)


class Layer:
    """A single fully-connected layer."""

    def __init__(self, input_size, output_size, activation="relu", seed=None):
        rng = random.Random(seed)
        scale = math.sqrt(2.0 / input_size)
        self.weights = [
            [rng.gauss(0, scale) for _ in range(input_size)]
            for _ in range(output_size)
        ]
        self.biases = [0.0] * output_size
        self.activation = activation
        self._last_z = []
        self._last_input = []

    def forward(self, x):
        """Compute layer output for input vector *x*."""
        self._last_input = list(x)
        z = [_dot(w, x) + b for w, b in zip(self.weights, self.biases)]
        self._last_z = z
        if self.activation == "sigmoid":
            return [_sigmoid(v) for v in z]
        return [_relu(v) for v in z]

    def backward(self, grad_output, lr):
        """Backprop through this layer; update weights; return grad_input."""
        if self.activation == "sigmoid":
            deriv = [_sigmoid_deriv(v) for v in self._last_z]
        else:
            deriv = [_relu_deriv(v) for v in self._last_z]

        delta = [g * d for g, d in zip(grad_output, deriv)]

        grad_input = [
            sum(self.weights[i][j] * delta[i] for i in range(len(delta)))
            for j in range(len(self._last_input))
        ]

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] -= lr * delta[i] * self._last_input[j]
            self.biases[i] -= lr * delta[i]

        return grad_input


class MLP:
    """Multi-layer perceptron for binary classification."""

    def __init__(self, layer_sizes, lr=0.01, seed=42):
        """Build the network.

        Args:
            layer_sizes: list of ints, e.g. [4, 8, 1] for a network with
                         4 inputs, one hidden layer of 8 units, and 1 output.
            lr: learning rate.
            seed: random seed for weight initialization.
        """
        self.lr = lr
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            activation = "sigmoid" if i == len(layer_sizes) - 2 else "relu"
            self.layers.append(
                Layer(layer_sizes[i], layer_sizes[i + 1], activation, seed=seed + i)
            )

    def predict_proba(self, x):
        """Return the scalar output probability for input vector *x*."""
        out = list(x)
        for layer in self.layers:
            out = layer.forward(out)
        return out[0]

    def predict(self, x, threshold=0.5):
        """Return binary prediction (0 or 1)."""
        return 1 if self.predict_proba(x) >= threshold else 0

    def train_step(self, x, y):
        """Single gradient-descent step for one sample.

        Args:
            x: input feature vector (list of floats).
            y: target label (0 or 1).

        Returns:
            binary cross-entropy loss for this sample.
        """
        prob = self.predict_proba(x)
        prob = max(1e-12, min(1 - 1e-12, prob))
        loss = -(y * math.log(prob) + (1 - y) * math.log(1 - prob))

        grad = [(prob - y)]
        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.lr)

        return loss
