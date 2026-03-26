"""Data loading and preprocessing utilities."""

import csv
import os
import random


def load_csv(filepath):
    """Load a CSV file and return a list of row dicts."""
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def train_test_split(data, test_ratio=0.2, seed=42):
    """Split data into training and test sets.

    Args:
        data: list of samples.
        test_ratio: fraction of data to use for testing.
        seed: random seed for reproducibility.

    Returns:
        tuple (train_data, test_data)
    """
    rng = random.Random(seed)
    shuffled = list(data)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def normalize(values):
    """Min-max normalize a list of numeric values to [0, 1].

    Args:
        values: list of floats.

    Returns:
        list of normalised floats.
    """
    if not values:
        return values
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return [0.0] * len(values)
    return [(v - min_v) / (max_v - min_v) for v in values]


def load_dataset(data_dir):
    """Load all CSV files from *data_dir* and return combined rows.

    Args:
        data_dir: path to directory containing .csv files.

    Returns:
        list of row dicts.
    """
    all_rows = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            all_rows.extend(load_csv(filepath))
    return all_rows
