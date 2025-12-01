"""
Plot GLUE benchmark metrics (accuracy, loss) across training rounds/epochs.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import torch
from tqdm import tqdm

# Import evaluation functions
from glue_eval import load_model_with_adapter, evaluate_sst2, evaluate_mrpc


def evaluate_model_on_glue(
    base_model: str,
    adapter_path: Optional[Path],
    device_map: str,
    dtype: str,
    max_samples: int,
    tasks: List[str],
) -> Dict:
    """Evaluate a model on GLUE tasks."""
    model, tokenizer = load_model_with_adapter(
        base_model,
        adapter_path,
        device_map,
        dtype
    )
    
    results = {}
    
    if "sst2" in tasks:
        sst2_result = evaluate_sst2(model, tokenizer, max_samples)
        results["sst2"] = sst2_result
    
    if "mrpc" in tasks:
        mrpc_result = evaluate_mrpc(model, tokenizer, max_samples)
        results["mrpc"] = mrpc_result
    
    # Clean up
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def find_round_adapters(adapters_root: Path, department: str = None) -> List[Dict]:
    """Find all adapter directories from different rounds."""
    adapters = []
    
    if department:
        # Look for specific department
        dept_path = adapters_root / department
        if dept_path.exists() and (dept_path / "adapter_config.json").exists():
            adapters.append({
                "path": dept_path,
                "department": department,
                "round": "final",  # Assume this is the final aggregated model
            })
    else:
        # Scan all departments
        for dept_dir in sorted(adapters_root.iterdir()):
            if dept_dir.is_dir() and (dept_dir / "adapter_config.json").exists():
                adapters.append({
                    "path": dept_dir,
                    "department": dept_dir.name,
                    "round": "final",
                })
    
    return adapters


def plot_metrics(
    metrics_data: Dict,
    output_dir: Path,
    title_prefix: str = "",
):
    """Plot accuracy and F1 metrics over rounds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for plotting
    tasks = list(metrics_data.keys())
    
    for task in tasks:
        task_data = metrics_data[task]
        rounds = sorted(task_data.keys())
        
        accuracies = [task_data[r]["accuracy"] for r in rounds]
        f1_scores = [task_data[r]["f1"] for r in rounds]
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{title_prefix}{task.upper()} Performance Over Rounds', fontsize=16, fontweight='bold')
        
        # Plot accuracy
        ax1.plot(rounds, accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Accuracy', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.0])
        
        # Add value labels on points
        for i, (r, acc) in enumerate(zip(rounds, accuracies)):
            ax1.annotate(f'{acc:.3f}', (r, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
        
        # Plot F1 score
        ax2.plot(rounds, f1_scores, marker='s', linewidth=2, markersize=8, color='#A23B72')
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('F1 Score', fontsize=12)
        ax2.set_title('F1 Score', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.0])
        
        # Add value labels on points
        for i, (r, f1) in enumerate(zip(rounds, f1_scores)):
            ax2.annotate(f'{f1:.3f}', (r, f1), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f"{task}_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {plot_path}")
        plt.close()
    
    # Create combined plot
    if len(tasks) > 1:
        fig, axes = plt.subplots(len(tasks), 2, figsize=(14, 5 * len(tasks)))
        fig.suptitle(f'{title_prefix}GLUE Benchmark Performance', fontsize=16, fontweight='bold')
        
        for idx, task in enumerate(tasks):
            task_data = metrics_data[task]
            rounds = sorted(task_data.keys())
            accuracies = [task_data[r]["accuracy"] for r in rounds]
            f1_scores = [task_data[r]["f1"] for r in rounds]
            
            # Accuracy subplot
            ax_acc = axes[idx, 0] if len(tasks) > 1 else axes[0]
            ax_acc.plot(rounds, accuracies, marker='o', linewidth=2, markersize=8)
            ax_acc.set_xlabel('Round', fontsize=12)
            ax_acc.set_ylabel('Accuracy', fontsize=12)
            ax_acc.set_title(f'{task.upper()} - Accuracy', fontsize=14)
            ax_acc.grid(True, alpha=0.3)
            ax_acc.set_ylim([0, 1.0])
            
            # F1 subplot
            ax_f1 = axes[idx, 1] if len(tasks) > 1 else axes[1]
            ax_f1.plot(rounds, f1_scores, marker='s', linewidth=2, markersize=8)
            ax_f1.set_xlabel('Round', fontsize=12)
            ax_f1.set_ylabel('F1 Score', fontsize=12)
            ax_f1.set_title(f'{task.upper()} - F1 Score', fontsize=14)
            ax_f1.grid(True, alpha=0.3)
            ax_f1.set_ylim([0, 1.0])
        
        plt.tight_layout()
        combined_path = output_dir / "combined_metrics.png"
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved combined plot: {combined_path}")
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot GLUE metrics over rounds")
    parser.add_argument("--base-model", default="Qwen/Qwen1.5-0.5B-Chat")
    parser.add_argument("--adapters-root", type=Path, default=Path("results/adapters"))
    parser.add_argument("--department", type=str, help="Evaluate specific department only")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tasks", nargs="+", default=["sst2", "mrpc"])
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    parser.add_argument("--baseline", action="store_true", 
                       help="Also evaluate baseline (no adapter)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Find adapters
    adapters = find_round_adapters(args.adapters_root, args.department)
    
    if not adapters and not args.baseline:
        print(f"❌ No adapters found in {args.adapters_root}")
        return 1
    
    print(f"Found {len(adapters)} adapter(s)")
    
    # Collect metrics across rounds
    metrics_data = {task: {} for task in args.tasks}
    
    # Evaluate baseline if requested
    if args.baseline:
        print("\n" + "="*70)
        print("Evaluating BASELINE (no adapter)")
        print("="*70)
        
        baseline_results = evaluate_model_on_glue(
            args.base_model,
            None,  # No adapter
            args.device_map,
            args.dtype,
            args.max_samples,
            args.tasks,
        )
        
        for task in args.tasks:
            if task in baseline_results:
                metrics_data[task][0] = baseline_results[task]
                print(f"{task.upper()}: Acc={baseline_results[task]['accuracy']:.4f}, F1={baseline_results[task]['f1']:.4f}")
    
    # Evaluate each adapter
    for idx, adapter_info in enumerate(adapters, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(adapters)}] Evaluating: {adapter_info['department']}")
        print(f"{'='*70}")
        
        results = evaluate_model_on_glue(
            args.base_model,
            adapter_info["path"],
            args.device_map,
            args.dtype,
            args.max_samples,
            args.tasks,
        )
        
        # Use round number or department index as x-axis
        round_num = idx if adapter_info['round'] == 'final' else int(adapter_info['round'])
        
        for task in args.tasks:
            if task in results:
                metrics_data[task][round_num] = results[task]
                print(f"{task.upper()}: Acc={results[task]['accuracy']:.4f}, F1={results[task]['f1']:.4f}")
    
    # Save metrics to JSON
    metrics_file = args.output_dir / "metrics_over_rounds.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\n✓ Saved metrics to {metrics_file}")
    
    # Plot metrics
    print("\nGenerating plots...")
    title_prefix = f"{args.department} - " if args.department else ""
    plot_metrics(metrics_data, args.output_dir, title_prefix)
    
    print(f"\n{'='*70}")
    print("✓ COMPLETE! Plots saved to:", args.output_dir)
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
