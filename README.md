# Base Paper Implementation

A clean, dependency-free Python prototype for implementing and evaluating machine-learning research papers.

## Project structure

```
Base_paper_implementation/
├── main.py               # Entry point – trains & evaluates the model
├── requirements.txt      # Optional third-party packages
├── src/
│   ├── data_loader.py    # CSV loading, normalization, train/test split
│   ├── model.py          # Pure-Python MLP (feed-forward neural network)
│   ├── trainer.py        # Training loop & evaluation metrics
│   └── utils.py          # JSON result persistence and pretty-printing
├── tests/
│   ├── test_data_loader.py
│   ├── test_model.py
│   └── test_trainer.py
└── data/                 # Place your .csv dataset files here
```

## Quick start

```bash
# No installation required – uses the Python standard library only.
python main.py
```

The prototype runs a binary-classification experiment on a small synthetic
dataset (XOR pattern) and writes the evaluation metrics to
`outputs/results.json`.

## Running tests

```bash
python -m pytest tests/ -v
# or without pytest:
python tests/test_data_loader.py
python tests/test_model.py
python tests/test_trainer.py
```

## Using your own data

1. Place one or more `.csv` files in the `data/` directory.
2. Update `main.py` to call `load_dataset("data/")` instead of the synthetic
   data block.
3. Set `FEATURE_KEYS` and `LABEL_KEY` to match your column names.
4. Adjust `MLP(layer_sizes=[...])` to match your input dimension.

## Architecture overview

| Module | Responsibility |
|---|---|
| `data_loader` | Load CSVs, normalise features, split data |
| `model.MLP` | Pure-Python multi-layer perceptron with back-propagation |
| `trainer.train` | Epoch-level training loop with per-epoch loss reporting |
| `trainer.evaluate` | Accuracy, precision, recall, F1 |
| `utils` | Save/load JSON results, pretty-print metrics |