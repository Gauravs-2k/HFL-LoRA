from tqdm import tqdm
from app.evaluation.models.predictor import predict
from app.evaluation.metrics.accuracy import accuracy


def evaluate_single_model(model, tokenizer, dataset):
    """
    Evaluates a (model, tokenizer) pair on a dataset of:
        { "input": ..., "target": ... }
    Returns accuracy + predictions.
    """
    predictions = []
    targets = []

    for item in tqdm(dataset, desc="Evaluating", ncols=80):
        pred = predict(model, tokenizer, item["input"])
        predictions.append(pred)
        targets.append(item["target"])

    return {
        "accuracy": accuracy(predictions, targets),
        "predictions": predictions,
        "targets": targets,
    }
