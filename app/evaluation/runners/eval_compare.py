from app.evaluation.runners.eval_single import evaluate_single_model
from app.evaluation.models.loader import (
    load_model,
    locate_department_lora,
    locate_federated_round_adapter,
)


def compare_models(base_model_name, department, dataset, rounds=[0]):
    results = {}

    # Base model
    tok, mod = load_model(base_model_name)
    results["base"] = evaluate_single_model(mod, tok, dataset)

    # Centralized department LoRA
    dept_lora = locate_department_lora(department)
    if dept_lora:
        tok, mod = load_model(base_model_name, dept_lora)
        results["dept_lora"] = evaluate_single_model(mod, tok, dataset)

    # Federated rounds
    for r in rounds:
        fed_dir = locate_federated_round_adapter(department, r)
        if fed_dir:
            tok, mod = load_model(base_model_name, fed_dir)
            results[f"fed_round_{r}"] = evaluate_single_model(mod, tok, dataset)

    return results
