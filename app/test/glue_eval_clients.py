"""
Client-Side GLUE Benchmark Evaluation

Evaluates INDIVIDUAL CLIENT models (before aggregation) on GLUE benchmarks.
This shows how well each client's personalized model performs.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

# Import from the main glue_eval script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from glue_eval import load_model_with_adapter, evaluate_sst2, evaluate_mrpc


def find_client_adapters(client_exports_root: Path, department: str = None):
    """Find all client adapter directories."""
    client_adapters = []
    
    if department:
        dept_dir = client_exports_root / department
        if dept_dir.exists():
            for client_dir in sorted(dept_dir.iterdir()):
                if client_dir.is_dir() and "client_" in client_dir.name:
                    client_adapters.append({
                        "path": client_dir,
                        "department": department,
                        "name": client_dir.name
                    })
    else:
        # Scan all departments
        for dept_dir in sorted(client_exports_root.iterdir()):
            if dept_dir.is_dir():
                department = dept_dir.name
                for client_dir in sorted(dept_dir.iterdir()):
                    if client_dir.is_dir() and "client_" in client_dir.name:
                        client_adapters.append({
                            "path": client_dir,
                            "department": department,
                            "name": client_dir.name
                        })
    
    return client_adapters


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CLIENT models on GLUE")
    parser.add_argument("--base-model", default="Qwen/Qwen1.5-0.5B-Chat")
    parser.add_argument("--client-exports", type=Path, default=Path("results/client_exports"))
    parser.add_argument("--department", type=str, help="Evaluate only this department (optional)")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tasks", nargs="+", default=["sst2", "mrpc"])
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-clients", type=int, default=0, 
                       help="Max clients to evaluate (0=all)")
    parser.add_argument("--output", type=Path, default=Path("results/glue_client_results.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Find all client adapters
    client_adapters = find_client_adapters(args.client_exports, args.department)
    
    if not client_adapters:
        print(f"❌ No client adapters found in {args.client_exports}")
        if args.department:
            print(f"   Department filter: {args.department}")
        return 1
    
    print(f"Found {len(client_adapters)} client models")
    
    if args.max_clients > 0:
        client_adapters = client_adapters[:args.max_clients]
        print(f"Evaluating first {len(client_adapters)} clients")
    
    # Task functions
    task_functions = {
        "sst2": evaluate_sst2,
        "mrpc": evaluate_mrpc,
    }
    
    results = {
        "base_model": args.base_model,
        "num_clients_evaluated": len(client_adapters),
        "clients": []
    }
    
    # Evaluate each client
    for idx, client_info in enumerate(client_adapters, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(client_adapters)}] Evaluating: {client_info['department']}/{client_info['name']}")
        print(f"{'='*70}")
        
        # Load model with client adapter
        model, tokenizer = load_model_with_adapter(
            args.base_model,
            client_info["path"],
            args.device_map,
            args.dtype
        )
        
        client_result = {
            "department": client_info["department"],
            "client_name": client_info["name"],
            "adapter_path": str(client_info["path"]),
            "tasks": {}
        }
        
        # Evaluate on each task
        for task in args.tasks:
            if task in task_functions:
                print(f"\nEvaluating {task.upper()}...")
                task_result = task_functions[task](model, tokenizer, args.max_samples)
                client_result["tasks"][task] = task_result
                print(f"  Accuracy: {task_result['accuracy']:.4f}")
                print(f"  F1: {task_result['f1']:.4f}")
            else:
                print(f"Warning: Task '{task}' not implemented")
        
        results["clients"].append(client_result)
        
        # Clean up
        del model
        del tokenizer
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Compute aggregated statistics
    print(f"\n{'='*70}")
    print("COMPUTING AGGREGATE STATISTICS")
    print(f"{'='*70}")
    
    aggregate_stats = {}
    for task in args.tasks:
        if task in task_functions:
            accuracies = [c["tasks"][task]["accuracy"] for c in results["clients"] if task in c["tasks"]]
            f1_scores = [c["tasks"][task]["f1"] for c in results["clients"] if task in c["tasks"]]
            
            aggregate_stats[task] = {
                "mean_accuracy": float(np.mean(accuracies)),
                "std_accuracy": float(np.std(accuracies)),
                "mean_f1": float(np.mean(f1_scores)),
                "std_f1": float(np.std(f1_scores)),
                "min_accuracy": float(np.min(accuracies)),
                "max_accuracy": float(np.max(accuracies)),
            }
    
    results["aggregate_stats"] = aggregate_stats
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("CLIENT-SIDE GLUE BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"Clients Evaluated: {len(client_adapters)}")
    print(f"{'='*70}")
    
    for task, stats in aggregate_stats.items():
        print(f"\n{task.upper()} Results:")
        print(f"  Mean Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
        print(f"  Range: [{stats['min_accuracy']:.4f}, {stats['max_accuracy']:.4f}]")
        print(f"  Mean F1: {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f}")
    
    print(f"\n{'='*70}")
    
    # Print per-department breakdown
    print("\nPer-Department Breakdown:")
    print(f"{'='*70}")
    
    departments = {}
    for client in results["clients"]:
        dept = client["department"]
        if dept not in departments:
            departments[dept] = []
        departments[dept].append(client)
    
    for dept, clients in sorted(departments.items()):
        print(f"\n{dept.upper()} ({len(clients)} clients):")
        for task in args.tasks:
            if task in task_functions:
                accs = [c["tasks"][task]["accuracy"] for c in clients if task in c["tasks"]]
                if accs:
                    print(f"  {task.upper()}: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
