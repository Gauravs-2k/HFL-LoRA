import os
import json
import csv
from datetime import datetime

from app.evaluation.datasets.department_loader import load_department_dataset
from app.evaluation.runners.eval_single import evaluate_single_model
from app.evaluation.runners.eval_clients import evaluate_clients
from app.evaluation.visualization.accuracy_plots import plot_round_accuracy
from app.evaluation.models.loader import (
    load_model,
    locate_department_lora,
    locate_federated_round_adapter,
)
from app.utils.config import settings


SAVE_ROOT = "results/evaluation"

# Department → dataset mapping (your Option A)
DEPARTMENTS = {
    "finance": "app/dataset/dept/FINANCE_dept.jsonl",
    "hr": "app/dataset/dept/HR_dept.jsonl",
    "engineering": "app/dataset/dept/ENGINEERING_dept.jsonl",
    "it_support": "app/dataset/dept/IT_SUPPORT_dept.jsonl",
}

# HuggingFace models you want to test (from your list)
HF_MODELS_PER_DEPT = {
    "finance": [
        "Gaurav2k/qwen1.5-1.8b-chat-finance",
        "Gaurav2k/qwen2-0.5b-finance",
        "Gaurav2k/qwen-dept-lora-finance",
    ],
    "hr": [
        "Gaurav2k/qwen1.5-1.8b-chat-hr",
        "Gaurav2k/qwen2-0.5b-hr",
        "Gaurav2k/qwen-dept-lora-hr",
    ],
    "engineering": [
        "Gaurav2k/qwen1.5-1.8b-chat-engineering",
        "Gaurav2k/qwen2-0.5b-engineering",
        "Gaurav2k/qwen-dept-lora-engineering",
    ],
    "it_support": [
        "Gaurav2k/qwen1.5-1.8b-chat-it_support",
        "Gaurav2k/qwen2-0.5b-it-support",
        "Gaurav2k/qwen-dept-lora-it-support",
    ],
}


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


# ---------------- Federated ----------------
def evaluate_federated_rounds(base_model: str, department: str, dataset):
    federated_results = {}

    for r in range(0, 101):
        adapter = locate_federated_round_adapter(department, r)
        if not adapter:
            continue

        tok, mod = load_model(base_model, adapter)
        res = evaluate_single_model(mod, tok, dataset)
        federated_results[r] = res["accuracy"]
        print(f"    Fed round {r}: {res['accuracy']:.4f}")

    return federated_results


# ---------------- Hugging Face models ----------------
def evaluate_hf_models(department: str, dataset):
    models = HF_MODELS_PER_DEPT.get(department, [])
    results = {}

    if not models:
        print("  No HF models configured for this department.")
        return results

    print("  Evaluating HuggingFace models:")
    for repo in models:
        print(f"    HF: {repo}")
        try:
            tok, mod = load_model(repo, None)  # repo is full model id
            res = evaluate_single_model(mod, tok, dataset)
            results[repo] = {"accuracy": res["accuracy"]}
            print(f"      accuracy: {res['accuracy']:.4f}")
        except Exception as e:
            results[repo] = {"error": str(e)}
            print(f"      ERROR: {e}")

    return results


# ---------------- Per-department ----------------
def evaluate_department(dept: str, dataset_path: str):
    print(f"\n=== Evaluating department: {dept} ===")

    if not os.path.isfile(dataset_path):
        print(f"  Dataset not found: {dataset_path}")
        return

    dataset = load_department_dataset(dataset_path)
    if not dataset:
        print(f"  Dataset empty or could not be loaded: {dataset_path}")
        return

    print(f"  Loaded {len(dataset)} samples.")
    save_dir = os.path.join(SAVE_ROOT, dept)
    ensure_dir(save_dir)

    base_model = settings.LLM_MODEL

    # Base model
    print("  Base model...")
    tok, mod = load_model(base_model)
    base_res = evaluate_single_model(mod, tok, dataset)
    print(f"    Base accuracy: {base_res['accuracy']:.4f}")

    # Dept LoRA
    print("  Department LoRA (if exists)...")
    dept_lora = locate_department_lora(dept)
    if dept_lora:
        tok, mod = load_model(base_model, dept_lora)
        dept_res = evaluate_single_model(mod, tok, dataset)
        print(f"    Dept LoRA accuracy: {dept_res['accuracy']:.4f}")
    else:
        dept_res = None
        print("    No dept LoRA found.")

    # Federated rounds
    print("  Federated rounds (if any)...")
    federated = evaluate_federated_rounds(base_model, dept, dataset)

    # Client adapters
    print("  Client adapters (if any)...")
    clients = evaluate_clients(
        base_model_name=base_model,
        department=dept,
        dataset=dataset,
        clients_root="client_adapters",
    )

    # HF models
    hf_res = evaluate_hf_models(dept, dataset)

    # JSON save
    out = {
        "department": dept,
        "dataset_path": dataset_path,
        "base": base_res,
        "dept_lora": dept_res,
        "federated": federated,
        "clients": clients,
        "huggingface_models": hf_res,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    json_path = os.path.join(save_dir, f"{dept}_eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved JSON → {json_path}")

    # CSV save
    csv_path = os.path.join(save_dir, f"{dept}_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy_or_info"])

        w.writerow(["base", base_res["accuracy"]])
        if dept_res:
            w.writerow(["dept_lora", dept_res["accuracy"]])

        for r, acc in sorted(federated.items()):
            w.writerow([f"fed_round_{r}", acc])

        for client_id, res in clients.items():
            w.writerow([f"client_{client_id}", res.get("accuracy", "NA")])

        for repo_id, res in hf_res.items():
            if "accuracy" in res:
                w.writerow([f"hf::{repo_id}", res["accuracy"]])
            else:
                w.writerow([f"hf::{repo_id}", f"ERROR: {res.get('error')}"])

    print(f"  Saved CSV → {csv_path}")

    # Plot federated accuracy if we have any
    if federated:
        try:
            plot_round_accuracy(federated, f"{dept.upper()} Federated Accuracy")
        except Exception as e:
            print(f"  Plotting error (ignored): {e}")

    print("  Done.\n")


def main():
    print("=== Running FULL EVALUATION (local + HF models) ===")
    ensure_dir(SAVE_ROOT)

    for dept, path in DEPARTMENTS.items():
        evaluate_department(dept, path)

    print("\n=== ALL DEPARTMENTS EVALUATED ===")


if __name__ == "__main__":
    main()
