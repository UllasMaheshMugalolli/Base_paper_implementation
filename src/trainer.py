"""Training loop and evaluation utilities."""


def train(model, train_data, feature_keys, label_key, epochs=10):
    """Train *model* on *train_data* for the specified number of epochs.

    Args:
        model: object with a ``train_step(x, y)`` method.
        train_data: list of dicts, each containing feature and label keys.
        feature_keys: ordered list of feature column names.
        label_key: name of the label column.
        epochs: number of passes over the training data.

    Returns:
        list of per-epoch average losses.
    """
    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for sample in train_data:
            x = [float(sample[k]) for k in feature_keys]
            y = float(sample[label_key])
            loss = model.train_step(x, y)
            total_loss += loss
        avg_loss = total_loss / max(len(train_data), 1)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")
    return epoch_losses


def evaluate(model, test_data, feature_keys, label_key, threshold=0.5):
    """Evaluate *model* on *test_data* and return accuracy and metrics dict.

    Args:
        model: object with a ``predict(x, threshold)`` method.
        test_data: list of dicts.
        feature_keys: ordered list of feature column names.
        label_key: name of the label column.
        threshold: classification threshold.

    Returns:
        dict with keys ``accuracy``, ``precision``, ``recall``, ``f1``.
    """
    tp = fp = tn = fn = 0
    for sample in test_data:
        x = [float(sample[k]) for k in feature_keys]
        y = int(float(sample[label_key]))
        pred = model.predict(x, threshold)
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 1:
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
