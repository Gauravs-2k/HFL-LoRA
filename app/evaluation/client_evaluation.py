import argparse
import json
from pathlib import Path
from typing import List

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

def discover_client_datasets(department: str, limit: int = -1) -> List[Path]:
    """
    Discover client datasets in the same order as department_client.py
    """
    root = Path("app/dataset") / f"{department}_personal_clients"
    candidates = sorted(root.glob("*.jsonl")) if root.exists() else []
    
    if not candidates:
        raise FileNotFoundError(f"No client data files found for department '{department}'")
    
    if limit > 0:
        return candidates[:limit]
    return candidates

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
        
        # Discover client adapter directories (they are named like round_1_client_0, round_1_client_1, etc.)
        client_dirs = sorted([d for d in dept_dir.iterdir() if d.is_dir() and d.name.startswith("round_") and "client_" in d.name])
        
        # Group by round
        rounds_dict = {}
        for client_dir in client_dirs:
            # Parse round number from directory name like "round_1_client_0"
            parts = client_dir.name.split('_')
            if len(parts) >= 3 and parts[0] == 'round' and parts[2] == 'client':
                round_num = f"round_{parts[1]}"
                client_idx = int(parts[3])
                
                if round_num not in rounds_dict:
                    rounds_dict[round_num] = []
                rounds_dict[round_num].append((client_idx, str(client_dir)))
        
        # Sort clients within each round and create the config
        for round_name, client_list in rounds_dict.items():
            client_list.sort(key=lambda x: x[0])  # Sort by client index
            client_paths = [path for _, path in client_list]
            config[dept_name][round_name] = client_paths
    
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
    parser.add_argument("--output", default="results/client_eval_all_rounds.json")
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
        "evaluation_type": "client_specific_datasets",
        "client_results": {}
    }

    for dept in config.keys():
        try:
            # Discover client datasets for this department
            client_datasets = discover_client_datasets(dept)
            print(f"Found {len(client_datasets)} client datasets for {dept}")
            
            print(f"\n{'='*60}")
            print(f"{dept.upper()} :: Evaluating clients on their own training data")
            print(f"{'='*60}")
            
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
                    
                    # Use the client's own training data for evaluation
                    if idx < len(client_datasets):
                        client_dataset_path = client_datasets[idx]
                        try:
                            dataset = load_department_dataset(str(client_dataset_path), max_samples=args.max_samples)
                            if not dataset:
                                print(f"    Client {idx}: EMPTY DATASET ({client_dataset_path})")
                                all_results["client_results"][dept][round_name].append({
                                    "client_id": idx,
                                    "path": client_path,
                                    "dataset_path": str(client_dataset_path),
                                    "error": "Empty dataset"
                                })
                                continue
                        except Exception as e:
                            print(f"    Client {idx}: DATASET LOAD FAILED ({client_dataset_path}): {e}")
                            all_results["client_results"][dept][round_name].append({
                                "client_id": idx,
                                "path": client_path,
                                "dataset_path": str(client_dataset_path),
                                "error": f"Dataset load failed: {e}"
                            })
                            continue
                    else:
                        print(f"    Client {idx}: NO CORRESPONDING DATASET (client {idx} >= {len(client_datasets)} datasets)")
                        all_results["client_results"][dept][round_name].append({
                            "client_id": idx,
                            "path": client_path,
                            "error": f"No corresponding dataset (client {idx} >= {len(client_datasets)} datasets)"
                        })
                        continue
                    
                    try:
                        client_summary = evaluate_client(base_model, client_path, dataset, args.dtype, args.device_map, dept)
                        result = {
                            "client_id": idx,
                            "path": client_path,
                            "dataset_path": str(client_dataset_path),
                            "samples": len(dataset),
                            "accuracy": client_summary["accuracy"]
                        }
                        all_results["client_results"][dept][round_name].append(result)
                        print(f"    Client {idx}: {client_summary['accuracy']:.4f} ({len(dataset)} samples from {client_dataset_path.name})")
                    except Exception as exc:
                        print(f"    Client {idx}: FAILED - {exc}")
                        all_results["client_results"][dept][round_name].append({
                            "client_id": idx,
                            "path": client_path,
                            "dataset_path": str(client_dataset_path),
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
# source env/bin/activate && PYTHONPATH=$PWD python app/evaluation/client_evaluation.py --max-samples 10 --device-map auto --dtype auto --rounds round_1 --output results/client_specific_eval.json