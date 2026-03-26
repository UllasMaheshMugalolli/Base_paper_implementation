"""General-purpose utility helpers."""

import json
import os


def save_results(results, filepath):
    """Persist a results dict to a JSON file.

    Args:
        results: dict of metric names to values.
        filepath: destination file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")


def load_results(filepath):
    """Load a previously saved results JSON file.

    Args:
        filepath: path to the JSON file.

    Returns:
        dict of results.
    """
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def print_metrics(metrics):
    """Pretty-print an evaluation metrics dict."""
    print("-" * 30)
    for key, value in metrics.items():
        print(f"  {key:<12}: {value:.4f}")
    print("-" * 30)
