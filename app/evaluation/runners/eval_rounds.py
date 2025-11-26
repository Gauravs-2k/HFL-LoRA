from app.evaluation.models.loader import (
    load_model,
    locate_federated_round_adapter
)
from app.evaluation.runners.eval_single import evaluate_single_model


def evaluate_rounds(base_model_name, department, dataset, rounds):
    """
    Evaluate federated model over multiple rounds.
    Returns { round_num: accuracy }
    """
    accs = {}

    for r in rounds:
        adapter = locate_federated_round_adapter(department, r)
        if not adapter:
            continue

        tok, mod = load_model(base_model_name, adapter)
        res = evaluate_single_model(mod, tok, dataset)
        accs[r] = res["accuracy"]

    return accs
