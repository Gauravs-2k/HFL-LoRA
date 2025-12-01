"""
Evaluate and plot GLUE performance for each department across training rounds.
Shows how federated learning improves each department over time.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import re

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch
from tqdm import tqdm

from glue_eval import load_model_with_adapter, evaluate_sst2, evaluate_mrpc


def find_department_rounds(results_dir: Path) -> Dict[str, List[Dict]]:
    """
    Find all department adapters organized by round.
    
    Returns:
        dict: {department: [{round: 1, path: Path}, {round: 2, path: Path}, ...]}
    """
    departments_data = {}
    
    # Check for round-based structure: results/adapters_round_N/department/
    for item in sorted(results_dir.iterdir()):
        if item.is_dir():
            # Check if this is a round directory (e.g., "adapters_round_1")
            round_match = re.search(r'round[_-]?(\d+)', item.name, re.IGNORECASE)
            
            if round_match:
                round_num = int(round_match.group(1))
                # Scan for departments in this round
                for dept_dir in sorted(item.iterdir()):
                    if dept_dir.is_dir() and (dept_dir / "adapter_config.json").exists():
                        dept_name = dept_dir.name
                        if dept_name not in departments_data:
                            departments_data[dept_name] = []
                        departments_data[dept_name].append({
                            "round": round_num,
                            "path": dept_dir
                        })
            # Also check in adapters/ for final models
            elif item.name == "adapters":
                for dept_dir in sorted(item.iterdir()):
                    if dept_dir.is_dir() and (dept_dir / "adapter_config.json").exists():
                        dept_name = dept_dir.name
                        if dept_name not in departments_data:
                            departments_data[dept_name] = []
                        # Assume this is the final round
                        max_round = max([max([r["round"] for r in rounds], default=0) 
                                       for rounds in departments_data.values()], default=0)
                        departments_data[dept_name].append({
                            "round": max_round + 1,
                            "path": dept_dir
                        })
    
    # Sort each department's rounds
    for dept in departments_data:
        departments_data[dept] = sorted(departments_data[dept], key=lambda x: x["round"])
    
    return departments_data


def evaluate_department_round(
    base_model: str,
    adapter_path: Path,
    device_map: str,
    dtype: str,
    max_samples: int,
    tasks: List[str],
) -> Dict:
    """Evaluate a department's adapter on GLUE tasks."""
    model, tokenizer = load_model_with_adapter(
        base_model,
        adapter_path,
        device_map,
        dtype
    )
    
    results = {}
    
    if "sst2" in tasks:
        results["sst2"] = evaluate_sst2(model, tokenizer, max_samples)
    
    if "mrpc" in tasks:
        results["mrpc"] = evaluate_mrpc(model, tokenizer, max_samples)
    
    # Clean up
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def plot_department_metrics(
    dept_metrics: Dict[str, Dict],
    output_dir: Path,
    baseline_metrics: Dict = None,
):
    """
    Plot accuracy and F1 for each department across rounds.
    
    Args:
        dept_metrics: {department: {round: {task: {accuracy, f1}}}}
        output_dir: Where to save plots
        baseline_metrics: Optional baseline (round 0) metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    departments = sorted(dept_metrics.keys())
    tasks = list(next(iter(next(iter(dept_metrics.values())).values())).keys())
    
    for task in tasks:
        # Create figure with subplots for each metric
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{task.upper()} Performance Across Departments and Rounds', 
                    fontsize=16, fontweight='bold')
        
        colors = plt.cm.Set2(range(len(departments)))
        
        for idx, dept in enumerate(departments):
            rounds = sorted(dept_metrics[dept].keys())
            accuracies = [dept_metrics[dept][r][task]["accuracy"] for r in rounds]
            f1_scores = [dept_metrics[dept][r][task]["f1"] for r in rounds]
            
            # Add baseline if provided
            if baseline_metrics and task in baseline_metrics:
                rounds = [0] + rounds
                accuracies = [baseline_metrics[task]["accuracy"]] + accuracies
                f1_scores = [baseline_metrics[task]["f1"]] + f1_scores
            
            # Plot accuracy
            ax1.plot(rounds, accuracies, marker='o', linewidth=2.5, markersize=8,
                    label=dept.capitalize(), color=colors[idx], alpha=0.8)
            
            # Plot F1
            ax2.plot(rounds, f1_scores, marker='s', linewidth=2.5, markersize=8,
                    label=dept.capitalize(), color=colors[idx], alpha=0.8)
        
        # Configure accuracy plot
        ax1.set_xlabel('Training Round', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
        ax1.set_title('Accuracy Over Rounds', fontsize=14)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='best', fontsize=11)
        ax1.set_ylim([0, 1.0])
        
        # Configure F1 plot
        ax2.set_xlabel('Training Round', fontsize=13, fontweight='bold')
        ax2.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
        ax2.set_title('F1 Score Over Rounds', fontsize=14)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='best', fontsize=11)
        ax2.set_ylim([0, 1.0])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f"{task}_departments_over_rounds.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        plt.close()
    
    # Create combined plot with all tasks
    if len(tasks) == 2:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('GLUE Benchmark Performance: All Departments Across Rounds', 
                    fontsize=18, fontweight='bold')
        
        colors = plt.cm.Set2(range(len(departments)))
        
        for task_idx, task in enumerate(tasks):
            for idx, dept in enumerate(departments):
                rounds = sorted(dept_metrics[dept].keys())
                accuracies = [dept_metrics[dept][r][task]["accuracy"] for r in rounds]
                f1_scores = [dept_metrics[dept][r][task]["f1"] for r in rounds]
                
                if baseline_metrics and task in baseline_metrics:
                    rounds = [0] + rounds
                    accuracies = [baseline_metrics[task]["accuracy"]] + accuracies
                    f1_scores = [baseline_metrics[task]["f1"]] + f1_scores
                
                # Accuracy
                axes[task_idx, 0].plot(rounds, accuracies, marker='o', linewidth=2,
                                      markersize=7, label=dept.capitalize(), 
                                      color=colors[idx], alpha=0.8)
                # F1
                axes[task_idx, 1].plot(rounds, f1_scores, marker='s', linewidth=2,
                                      markersize=7, label=dept.capitalize(),
                                      color=colors[idx], alpha=0.8)
            
            # Configure plots
            axes[task_idx, 0].set_xlabel('Round', fontsize=12)
            axes[task_idx, 0].set_ylabel('Accuracy', fontsize=12)
            axes[task_idx, 0].set_title(f'{task.upper()} - Accuracy', fontsize=13, fontweight='bold')
            axes[task_idx, 0].grid(True, alpha=0.3, linestyle='--')
            axes[task_idx, 0].legend(loc='best', fontsize=10)
            axes[task_idx, 0].set_ylim([0, 1.0])
            
            axes[task_idx, 1].set_xlabel('Round', fontsize=12)
            axes[task_idx, 1].set_ylabel('F1 Score', fontsize=12)
            axes[task_idx, 1].set_title(f'{task.upper()} - F1 Score', fontsize=13, fontweight='bold')
            axes[task_idx, 1].grid(True, alpha=0.3, linestyle='--')
            axes[task_idx, 1].legend(loc='best', fontsize=10)
            axes[task_idx, 1].set_ylim([0, 1.0])
        
        plt.tight_layout()
        combined_path = output_dir / "all_departments_all_tasks.png"
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {combined_path}")
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate department adapters on GLUE across rounds and plot results"
    )
    parser.add_argument("--base-model", default="Qwen/Qwen1.5-0.5B-Chat")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tasks", nargs="+", default=["sst2", "mrpc"])
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("results/department_plots"))
    parser.add_argument("--baseline", action="store_true", help="Evaluate baseline (no adapter)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Find all department adapters across rounds
    print("Scanning for department adapters...")
    dept_rounds = find_department_rounds(args.results_dir)
    
    if not dept_rounds:
        print(f"❌ No department adapters found in {args.results_dir}")
        print("   Make sure your training has completed and saved adapters!")
        return 1
    
    print(f"\nFound departments:")
    for dept, rounds in dept_rounds.items():
        print(f"  {dept}: {len(rounds)} rounds ({[r['round'] for r in rounds]})")
    
    # Evaluate baseline if requested
    baseline_metrics = None
    if args.baseline:
        print("\n" + "="*70)
        print("EVALUATING BASELINE (No Adapter)")
        print("="*70)
        
        baseline_metrics = {}
        model, tokenizer = load_model_with_adapter(
            args.base_model, None, args.device_map, args.dtype
        )
        
        if "sst2" in args.tasks:
            baseline_metrics["sst2"] = evaluate_sst2(model, tokenizer, args.max_samples)
            print(f"SST2: Acc={baseline_metrics['sst2']['accuracy']:.4f}, F1={baseline_metrics['sst2']['f1']:.4f}")
        
        if "mrpc" in args.tasks:
            baseline_metrics["mrpc"] = evaluate_mrpc(model, tokenizer, args.max_samples)
            print(f"MRPC: Acc={baseline_metrics['mrpc']['accuracy']:.4f}, F1={baseline_metrics['mrpc']['f1']:.4f}")
        
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Evaluate each department at each round
    dept_metrics = {}
    
    for dept, rounds_list in dept_rounds.items():
        dept_metrics[dept] = {}
        
        for round_info in tqdm(rounds_list, desc=f"{dept.capitalize()}", unit="round"):
            round_num = round_info["round"]
            adapter_path = round_info["path"]
            
            print(f"\n{'='*70}")
            print(f"Evaluating: {dept} - Round {round_num}")
            print(f"{'='*70}")
            
            results = evaluate_department_round(
                args.base_model,
                adapter_path,
                args.device_map,
                args.dtype,
                args.max_samples,
                args.tasks,
            )
            
            dept_metrics[dept][round_num] = results
            
            for task in args.tasks:
                if task in results:
                    print(f"{task.upper()}: Acc={results[task]['accuracy']:.4f}, F1={results[task]['f1']:.4f}")
    
    # Save metrics to JSON
    metrics_file = args.output_dir / "department_metrics_by_round.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    save_data = {
        "baseline": baseline_metrics,
        "departments": {
            dept: {
                str(round_num): round_data
                for round_num, round_data in rounds.items()
            }
            for dept, rounds in dept_metrics.items()
        }
    }
    
    with open(metrics_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n✓ Saved metrics to {metrics_file}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_department_metrics(dept_metrics, args.output_dir, baseline_metrics)
    
    print(f"\n{'='*70}")
    print("✓ COMPLETE!")
    print(f"Plots saved to: {args.output_dir}")
    print(f"Metrics saved to: {metrics_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
