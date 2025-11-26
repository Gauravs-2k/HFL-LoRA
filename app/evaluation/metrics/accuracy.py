def accuracy(predictions, targets):
    """
    Simple accuracy metric for classification-like scenarios.
    Predictions and targets should both be lists of strings.
    """
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(targets) if targets else 0
