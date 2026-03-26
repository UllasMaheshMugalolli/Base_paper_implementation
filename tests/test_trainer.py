"""Tests for trainer module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import MLP
from src.trainer import evaluate, train


def _make_data():
    return [
        {"x1": "0.0", "x2": "0.0", "label": "0"},
        {"x1": "0.0", "x2": "1.0", "label": "1"},
        {"x1": "1.0", "x2": "0.0", "label": "1"},
        {"x1": "1.0", "x2": "1.0", "label": "0"},
    ] * 10


def test_train_returns_losses():
    model = MLP(layer_sizes=[2, 4, 1], lr=0.05, seed=0)
    data = _make_data()
    losses = train(model, data, ["x1", "x2"], "label", epochs=3)
    assert len(losses) == 3
    assert all(l >= 0 for l in losses)


def test_evaluate_returns_metrics():
    model = MLP(layer_sizes=[2, 4, 1], lr=0.1, seed=0)
    data = _make_data()
    train(model, data, ["x1", "x2"], "label", epochs=50)
    metrics = evaluate(model, data, ["x1", "x2"], "label")
    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_evaluate_perfect():
    """A model that always predicts 1 on all-positive data."""

    class AlwaysOne:
        def predict(self, x, threshold=0.5):
            return 1

    data = [{"f": "1.0", "label": "1"}] * 10
    metrics = evaluate(AlwaysOne(), data, ["f"], "label")
    assert metrics["accuracy"] == 1.0
    assert metrics["recall"] == 1.0


if __name__ == "__main__":
    test_train_returns_losses()
    test_evaluate_returns_metrics()
    test_evaluate_perfect()
    print("All trainer tests passed.")
