"""Tests for model module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import MLP, Layer, _sigmoid, _relu


def test_sigmoid_range():
    for z in [-10.0, -1.0, 0.0, 1.0, 10.0]:
        s = _sigmoid(z)
        assert 0.0 < s < 1.0, f"sigmoid({z}) = {s} out of (0,1)"


def test_relu():
    assert _relu(-5.0) == 0.0
    assert _relu(0.0) == 0.0
    assert _relu(3.0) == 3.0


def test_layer_forward_output_size():
    layer = Layer(input_size=4, output_size=3, activation="relu", seed=7)
    out = layer.forward([1.0, 2.0, 3.0, 4.0])
    assert len(out) == 3


def test_mlp_predict_binary():
    model = MLP(layer_sizes=[2, 4, 1], lr=0.01, seed=0)
    pred = model.predict([0.5, 0.5])
    assert pred in (0, 1)


def test_mlp_predict_proba_range():
    model = MLP(layer_sizes=[3, 6, 1], lr=0.01, seed=1)
    prob = model.predict_proba([0.1, 0.9, 0.5])
    assert 0.0 <= prob <= 1.0


def test_mlp_train_step_returns_loss():
    model = MLP(layer_sizes=[2, 4, 1], lr=0.1, seed=5)
    loss = model.train_step([0.0, 1.0], 1)
    assert loss >= 0.0


def test_mlp_loss_decreases():
    """Loss should trend downward over many steps on a simple sample."""
    model = MLP(layer_sizes=[2, 8, 1], lr=0.1, seed=42)
    x, y = [1.0, 0.0], 1
    losses = [model.train_step(x, y) for _ in range(200)]
    assert losses[-1] < losses[0], "Loss did not decrease after 200 steps"


if __name__ == "__main__":
    test_sigmoid_range()
    test_relu()
    test_layer_forward_output_size()
    test_mlp_predict_binary()
    test_mlp_predict_proba_range()
    test_mlp_train_step_returns_loss()
    test_mlp_loss_decreases()
    print("All model tests passed.")
