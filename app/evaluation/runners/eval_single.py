from tqdm import tqdm
from app.evaluation.models.predictor import predict
from app.evaluation.metrics.accuracy import accuracy
from app.evaluation.metrics.eng_accuracy import accuracy as eng_accuracy


def evaluate_single_model(model, tokenizer, dataset, accuracy_func=None):
    if accuracy_func is None:
        accuracy_func = accuracy
    predictions = []
    targets = []

    for item in tqdm(dataset, desc="Evaluating", ncols=80):
        pred = predict(model, tokenizer, item["input"])
        predictions.append(pred)
        targets.append(item["target"])

    return {
        "accuracy": accuracy_func(predictions, targets),
        "predictions": predictions,
        "targets": targets,
    }
