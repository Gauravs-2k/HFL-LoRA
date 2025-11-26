import os
from app.evaluation.models.loader import load_model
from app.evaluation.runners.eval_single import evaluate_single_model


def evaluate_clients(base_model_name, department, dataset, clients_root="client_adapters"):
    """
    Evaluate multiple client adapters:
        client_adapters/<department>/<client_id>/
    """
    dept_root = os.path.join(clients_root, department)
    if not os.path.isdir(dept_root):
        return {}

    results = {}

    for client_id in os.listdir(dept_root):
        adapter_dir = os.path.join(dept_root, client_id)
        if not os.path.isdir(adapter_dir):
            continue

        tok, mod = load_model(base_model_name, adapter_dir)
        res = evaluate_single_model(mod, tok, dataset)
        results[client_id] = res

    return results
