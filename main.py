"""Entry point for the prototype experiment."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import normalize, train_test_split
from src.model import MLP
from src.trainer import evaluate, train
from src.utils import print_metrics, save_results

# ---------------------------------------------------------------------------
# Tiny synthetic dataset (XOR-style binary classification) so the prototype
# runs without any external files.
# ---------------------------------------------------------------------------
SYNTHETIC_DATA = [
    {"x1": 0.0, "x2": 0.0, "label": 0},
    {"x1": 0.0, "x2": 1.0, "label": 1},
    {"x1": 1.0, "x2": 0.0, "label": 1},
    {"x1": 1.0, "x2": 1.0, "label": 0},
] * 50  # replicate to give the trainer enough samples


FEATURE_KEYS = ["x1", "x2"]
LABEL_KEY = "label"


def main():
    print("=== Base Paper Implementation Prototype ===\n")

    # --- Data preparation --------------------------------------------------
    data = []
    x1_vals = [float(d["x1"]) for d in SYNTHETIC_DATA]
    x2_vals = [float(d["x2"]) for d in SYNTHETIC_DATA]
    x1_norm = normalize(x1_vals)
    x2_norm = normalize(x2_vals)
    for i, d in enumerate(SYNTHETIC_DATA):
        data.append({"x1": x1_norm[i], "x2": x2_norm[i], "label": d["label"]})

    train_data, test_data = train_test_split(data, test_ratio=0.2, seed=42)
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}\n")

    # --- Model -------------------------------------------------------------
    model = MLP(layer_sizes=[2, 8, 1], lr=0.1, seed=0)

    # --- Training ----------------------------------------------------------
    epoch_losses = train(
        model, train_data, FEATURE_KEYS, LABEL_KEY, epochs=100
    )
    print()

    # --- Evaluation --------------------------------------------------------
    metrics = evaluate(model, test_data, FEATURE_KEYS, LABEL_KEY)
    print("Evaluation results:")
    print_metrics(metrics)

    # --- Persist results ---------------------------------------------------
    results = {
        "final_train_loss": epoch_losses[-1],
        **metrics,
    }
    save_results(results, "outputs/results.json")


if __name__ == "__main__":
    main()
