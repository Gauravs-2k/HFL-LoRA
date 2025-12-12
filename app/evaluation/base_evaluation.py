import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.evaluation.datasets.department_loader import load_department_dataset
from app.evaluation.models.loader import load_model
from app.evaluation.runners.eval_single import evaluate_single_model
from app.evaluation.metrics.eng_accuracy import accuracy as eng_accuracy
from app.evaluation.metrics.accuracy import accuracy as simple_accuracy
from app.model.inference import parse_dtype
from app.utils.config import settings

TEST_FILES = {
    "finance": "FINANCE_test.jsonl",
    "hr": "HR_test.jsonl", 
    "engineering": "ENGINEERING_test.jsonl",
    "it_support": "IT_SUPPORT_test.jsonl",
    "customer_support": "IT_SUPPORT_test.jsonl",  # customer_support uses IT_SUPPORT test data
}

def discover_clients_from_results(results_root: Path = Path("results")):
    """
    Automatically discover client adapters from the results folder structure.
    Returns a dict similar to client_lora_federated.json format.
    """
    client_exports = results_root / "client_exports"
    if not client_exports.exists():
        raise FileNotFoundError(f"Client exports not found at {client_exports}")
    
    config = {}
    
    # Discover departments
    for dept_dir in client_exports.iterdir():
        if not dept_dir.is_dir():
            continue
        
        dept_name = dept_dir.name
        config[dept_name] = {}
        
        # Discover rounds within each department
        round_dirs = sorted([d for d in dept_dir.iterdir() if d.is_dir() and d.name.startswith("round_")])
        
        for round_dir in round_dirs:
            round_name = round_dir.name  # e.g., "round_1"
            client_adapters = []
            
            # Find all client adapter directories in this round
            client_dirs = sorted([d for d in round_dir.iterdir() if d.is_dir() and d.name.startswith("round_") and "client_" in d.name])
            if client_dirs:
                # Sort by client index
                client_dirs.sort(key=lambda x: int(x.name.split('_')[-1]))
                client_adapters = [str(d) for d in client_dirs]
            
            if client_adapters:  # Only add rounds that have adapters
                config[dept_name][round_name] = client_adapters
    
    if not config:
        raise ValueError(f"No client adapters found in {client_exports}")
    
    return config

def resolve_base_model(value):
    if value:
        return value
    return "Qwen/Qwen1.5-1.8B-Chat"

def dataset_path(root, department):
    filename = TEST_FILES.get(department)
    if not filename:
        raise ValueError(f"Unsupported department {department}")
    return Path(root) / filename

def load_pair(base_model, candidate, dtype, device_map):
    if not candidate:
        return load_model(base_model, None, dtype, device_map)
    path = Path(candidate)
    if path.exists():
        return load_model(base_model, str(path), dtype, device_map)
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_kwargs = {"trust_remote_code": True}
        torch_dtype = parse_dtype(dtype)
        if torch_dtype:
            model_kwargs["torch_dtype"] = torch_dtype
        if device_map.lower() != "none":
            model_kwargs["device_map"] = device_map
        if settings.HF_TOKEN:
            model_kwargs["token"] = settings.HF_TOKEN
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, candidate, is_trainable=False)
        model.eval()
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(f"Failed to load {candidate}: {e}")

def evaluate_client(base_model, client_path, dataset, dtype, device_map, dept):
    tokenizer, model = load_pair(base_model, client_path, dtype, device_map)
    accuracy_func = eng_accuracy if dept == "engineering" else simple_accuracy
    res = evaluate_single_model(model, tokenizer, dataset, accuracy_func)
    summary = {"accuracy": res.get("accuracy", 0.0)}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
  
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model")
    parser.add_argument("--config", help="Path to config JSON file (optional - will auto-discover if not provided)")
    parser.add_argument("--results-root", default="results", help="Root directory containing client_exports folder")
    parser.add_argument("--auto-discover", action="store_true", default=True, 
                       help="Automatically discover clients from results folder (default: True)")
    parser.add_argument("--dataset-root", default="app/dataset/test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--output", default="results/base_eval_all_rounds.json")
    parser.add_argument("--rounds", nargs="*", help="Specific rounds to evaluate (e.g., round_1 round_5)")
    args = parser.parse_args()

    base_model = resolve_base_model(args.base_model)
    
    # Auto-discover clients from results folder if no config provided
    if args.config and not args.auto_discover:
        config_path = Path(args.config)
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        print(f"Loaded config from: {config_path}")
    else:
        try:
            config = discover_clients_from_results(Path(args.results_root))
            print(f"Auto-discovered clients from: {args.results_root}/client_exports")
        except Exception as e:
            print(f"Failed to auto-discover clients: {e}")
            if args.config:
                print(f"Falling back to config file: {args.config}")
                config_path = Path(args.config)
                with open(config_path, "r", encoding="utf-8") as handle:
                    config = json.load(handle)
            else:
                raise

    all_results = {
        "base_model": base_model,
        "source": "auto_discovered" if args.auto_discover and not args.config else f"config_file: {args.config}",
        "results_root": args.results_root,
        "base_accuracies": {},
        "client_results": {}
    }

    for dept in config.keys():
        try:
            data_path = dataset_path(args.dataset_root, dept)
            if not data_path.exists():
                print(f"Dataset missing for {dept}: {data_path}")
                continue
            dataset = load_department_dataset(str(data_path), max_samples=args.max_samples)
            if not dataset:
                print(f"Dataset empty for {dept}: {data_path}")
                continue
            
            print(f"\n{'='*60}")
            print(f"{dept.upper()} :: samples={len(dataset)}")
            print(f"{'='*60}")
            
            base_summary = evaluate_client(base_model, None, dataset, args.dtype, args.device_map, dept)
            all_results["base_accuracies"][dept] = base_summary["accuracy"]
            print(f"  Base model accuracy: {base_summary['accuracy']:.4f}")
            
            all_results["client_results"][dept] = {}
            
            rounds_data = config[dept]
            selected_rounds = args.rounds if args.rounds else list(rounds_data.keys())
            
            for round_name in selected_rounds:
                if round_name not in rounds_data:
                    continue
                clients = rounds_data[round_name]
                all_results["client_results"][dept][round_name] = []
                
                print(f"\n  {round_name}:")
                for idx, client_path in enumerate(clients):
                    if not Path(client_path).exists():
                        print(f"    Client {idx}: MISSING ({client_path})")
                        all_results["client_results"][dept][round_name].append({
                            "client_id": idx,
                            "path": client_path,
                            "error": "Path not found"
                        })
                        continue
                    
                    try:
                        client_summary = evaluate_client(base_model, client_path, dataset, args.dtype, args.device_map, dept)
                        delta = client_summary["accuracy"] - base_summary["accuracy"]
                        result = {
                            "client_id": idx,
                            "path": client_path,
                            "accuracy": client_summary["accuracy"],
                            "delta_vs_base": delta
                        }
                        all_results["client_results"][dept][round_name].append(result)
                        print(f"    Client {idx}: {client_summary['accuracy']:.4f} (Δ {delta:+.4f})")
                    except Exception as exc:
                        print(f"    Client {idx}: FAILED - {exc}")
                        all_results["client_results"][dept][round_name].append({
                            "client_id": idx,
                            "path": client_path,
                            "error": str(exc)
                        })
                        
        except Exception as exc:
            print(f"Failed for {dept}: {exc}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)
    print(f"\n{'='*60}")
    print(f"All results saved to {output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    run()
    
# Usage examples:
# Auto-discover clients from results folder:
# source env/bin/activate && PYTHONPATH=$PWD python app/evaluation/base_evaluation.py --max-samples 50 --device-map auto --dtype auto