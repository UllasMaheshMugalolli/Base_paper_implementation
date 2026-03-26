"""Tests for data_loader module."""

import os
import sys
import tempfile
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_csv, normalize, train_test_split, load_dataset


def test_normalize_basic():
    values = [0.0, 5.0, 10.0]
    result = normalize(values)
    assert result == [0.0, 0.5, 1.0], f"Unexpected result: {result}"


def test_normalize_constant():
    values = [3.0, 3.0, 3.0]
    result = normalize(values)
    assert result == [0.0, 0.0, 0.0]


def test_normalize_empty():
    assert normalize([]) == []


def test_train_test_split_sizes():
    data = list(range(100))
    train, test = train_test_split(data, test_ratio=0.2, seed=0)
    assert len(train) == 80
    assert len(test) == 20


def test_train_test_split_no_overlap():
    data = list(range(50))
    train, test = train_test_split(data, test_ratio=0.3, seed=1)
    train_set = set(train)
    test_set = set(test)
    assert train_set.isdisjoint(test_set)
    assert train_set | test_set == set(data)


def test_load_csv(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")
    rows = load_csv(str(csv_file))
    assert len(rows) == 2
    assert rows[0] == {"a": "1", "b": "2", "c": "3"}
    assert rows[1] == {"a": "4", "b": "5", "c": "6"}


def test_load_dataset(tmp_path):
    (tmp_path / "file1.csv").write_text("x,y\n1,2\n3,4\n")
    (tmp_path / "file2.csv").write_text("x,y\n5,6\n")
    (tmp_path / "not_csv.txt").write_text("ignore me")
    rows = load_dataset(str(tmp_path))
    assert len(rows) == 3


if __name__ == "__main__":
    test_normalize_basic()
    test_normalize_constant()
    test_normalize_empty()
    test_train_test_split_sizes()
    test_train_test_split_no_overlap()

    import tempfile, pathlib
    test_load_csv(pathlib.Path(tempfile.mkdtemp()))
    test_load_dataset(pathlib.Path(tempfile.mkdtemp()))
    print("All data_loader tests passed.")
