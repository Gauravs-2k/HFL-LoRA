"""
Client-Side GLUE Benchmark Evaluation & Plotting

Evaluates INDIVIDUAL CLIENT models across rounds and plots performance progression.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from tqdm import tqdm

# Import from the main glue_eval script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from glue_eval import load_model_with_adapter, evaluate_sst2, evaluate_mrpc


def find_client_adapters(client_exports_root: Path, department: str = None):
    """
    Find all client adapter directories and parse round numbers.
    Returns list of dicts with path, department, name, round, client_id.
    """
    client_adapters = []
    
    # Helper to scan a department directory
    def scan_dept(dept_path):
        dept_name = dept_path.name
        for client_dir in sorted(dept_path.iterdir()):
            if client_dir.is_dir() and "client_" in client_dir.name:
                # Parse round and client ID
                # Format: round_1_client_0
                match = re.search(r'round_(\d+)_client_(\d+)', client_dir.name)
                if match:
                    round_num = int(match.group(1))
                    client_id = int(match.group(2))
                    
                    client_adapters.append({
                        "path": client_dir,
                        "department": dept_name,
                        "name": client_dir.name,
                        "round": round_num,
                        "client_id": client_id
                    })

    if department:
        dept_dir = client_exports_root / department
        if dept_dir.exists():
            scan_dept(dept_dir)
    else:
        # Scan all departments
        for dept_dir in sorted(client_exports_root.iterdir()):
            if dept_dir.is_dir():
                scan_dept(dept_dir)
    
    # Sort by round, then department, then client
    client_adapters.sort(key=lambda x: (x["round"], x["department"], x["client_id"]))
    return client_adapters


def plot_client_metrics(results: Dict, output_dir: Path):
    """Plot average client accuracy vs round."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Organize data by round
    rounds_data = {}  # {round: {task: [accuracies]}}
    
    for client in results["clients"]:
        r = client["round"]
        if r not in rounds_data:
            rounds_data[r] = {}
        
        for task, metrics in client["tasks"].items():
            if task not in rounds_data[r]:
                rounds_data[r][task] = []
            rounds_data[r][task].append(metrics["accuracy"])
    
    rounds = sorted(rounds_data.keys())
    tasks = list(rounds_data[rounds[0]].keys()) if rounds else []
    
    # Plot per task
    for task in tasks:
        plt.figure(figsize=(10, 6))
        
        means = []
        stds = []
        
        for r in rounds:
            accs = rounds_data[r].get(task, [])
            if accs:
                means.append(np.mean(accs))
                stds.append(np.std(accs))
            else:
                means.append(0)
                stds.append(0)
        
        plt.errorbar(rounds, means, yerr=stds, fmt='o-', linewidth=2, capsize=5, label='Mean Client Accuracy')
        
        plt.xlabel('Training Round (Epoch)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'{task.upper()} Client Accuracy vs Rounds', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim([0, 1.0])
        
        # Add value labels
        for i, val in enumerate(means):
            plt.annotate(f'{val:.3f}', (rounds[i], val), textcoords="offset points", xytext=(0,10), ha='center')
            
        plt.tight_layout()
        plt.savefig(output_dir / f"{task}_client_accuracy_vs_rounds.png", dpi=300)
        plt.close()
        print(f"✓ Saved plot: {output_dir / f'{task}_client_accuracy_vs_rounds.png'}")


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
    parser.add_argument("--round", type=int, default=None,
                       help="Evaluate only specific round (e.g. 9)")
    parser.add_argument("--output", type=Path, default=Path("results/glue_client_results.json"))
    parser.add_argument("--plot-dir", type=Path, default=Path("results/plots"))
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Find all client adapters
    client_adapters = find_client_adapters(args.client_exports, args.department)
    
    # Filter by round if specified
    if args.round is not None:
        print(f"Filtering for Round {args.round}...")
        client_adapters = [c for c in client_adapters if c["round"] == args.round]
    
    if not client_adapters:
        print(f"❌ No client adapters found in {args.client_exports}")
        if args.department:
            print(f"   Department filter: {args.department}")
        if args.round is not None:
            print(f"   Round filter: {args.round}")
        return 1
    
    print(f"Found {len(client_adapters)} client models across {len(set(c['round'] for c in client_adapters))} rounds")
    
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
        print(f"[{idx}/{len(client_adapters)}] Evaluating: {client_info['department']}/Round {client_info['round']}/Client {client_info['client_id']}")
        print(f"{'='*70}")
        
        # Load model with client adapter
        try:
            model, tokenizer = load_model_with_adapter(
                args.base_model,
                client_info["path"],
                args.device_map,
                args.dtype
            )
        except Exception as e:
            print(f"❌ Error loading adapter: {e}")
            continue
        
        client_result = {
            "department": client_info["department"],
            "client_name": client_info["name"],
            "round": client_info["round"],
            "client_id": client_info["client_id"],
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
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_client_metrics(results, args.plot_dir)
    
    print(f"\n{'='*70}")
    print("✓ COMPLETE!")
    print(f"Plots saved to: {args.plot_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
