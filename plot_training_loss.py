"""
Plot Training Loss Over Federated Learning Rounds

Generates a publication-quality plot of training loss progression.
"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path

# Sample training loss data (replace with actual data from your training logs)
# This simulates the typical federated learning loss curve
training_data = {
    "rounds": list(range(1, 11)),  # Rounds 1-10
    "departments": {
        "engineering": {
            "loss_per_round": [1.45, 1.12, 0.95, 0.82, 0.73, 0.67, 0.62, 0.58, 0.55, 0.53]
        },
        "finance": {
            "loss_per_round": [1.52, 1.18, 0.98, 0.84, 0.75, 0.69, 0.64, 0.60, 0.57, 0.55]
        },
        "hr": {
            "loss_per_round": [1.48, 1.15, 0.96, 0.83, 0.74, 0.68, 0.63, 0.59, 0.56, 0.54]
        },
        "customer_support": {
            "loss_per_round": [1.50, 1.16, 0.97, 0.85, 0.76, 0.70, 0.65, 0.61, 0.58, 0.56]
        }
    }
}

def plot_training_loss():
    """Create training loss plot."""
    
    rounds = training_data["rounds"]
    departments = training_data["departments"]
    
    # Create figure with larger size for publication
    plt.figure(figsize=(12, 7))
    
    # Color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    # Plot each department
    for idx, (dept, color) in enumerate(zip(departments.keys(), colors)):
        loss = departments[dept]["loss_per_round"]
        plt.plot(rounds, loss, marker='o', linewidth=2.5, markersize=8, 
                label=dept.replace('_', ' ').title(), color=color, alpha=0.85)
    
    # Calculate and plot average across departments
    avg_loss = []
    for i in range(len(rounds)):
        round_losses = [departments[dept]["loss_per_round"][i] for dept in departments]
        avg_loss.append(np.mean(round_losses))
    
    plt.plot(rounds, avg_loss, marker='s', linewidth=3, markersize=10,
            label='Global Average', color='black', linestyle='--', alpha=0.7)
    
    # Styling
    plt.xlabel('Training Round (Epoch)', fontsize=14, fontweight='bold')
    plt.ylabel('Training Loss', fontsize=14, fontweight='bold')
    plt.title('Federated Learning: Training Loss Progression', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
    
    # Set y-axis to start from 0 or adjust based on data
    plt.ylim([0, max(max(d["loss_per_round"]) for d in departments.values()) * 1.1])
    
    # Add value labels on key points
    for i in [0, len(rounds)-1]:  # First and last points
        plt.annotate(f'{avg_loss[i]:.2f}', 
                    (rounds[i], avg_loss[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=10,
                    fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("results/plots/training_loss_vs_rounds.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved training loss plot to: {output_path}")
    
    # Also create a combined metrics plot
    create_combined_metrics_plot(output_path.parent)


def create_combined_metrics_plot(output_dir):
    """Create a 2x2 grid showing loss, accuracy, and convergence."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Federated Learning Training Metrics', fontsize=18, fontweight='bold', y=0.995)
    
    rounds = training_data["rounds"]
    departments = training_data["departments"]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    for dept, color in zip(departments.keys(), colors):
        loss = departments[dept]["loss_per_round"]
        ax1.plot(rounds, loss, marker='o', linewidth=2, markersize=6, 
                label=dept.replace('_', ' ').title(), color=color, alpha=0.85)
    ax1.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Plot 2: Loss Improvement (%)
    ax2 = axes[0, 1]
    for dept, color in zip(departments.keys(), colors):
        loss = departments[dept]["loss_per_round"]
        improvement = [(loss[0] - l) / loss[0] * 100 for l in loss]
        ax2.plot(rounds, improvement, marker='s', linewidth=2, markersize=6,
                label=dept.replace('_', ' ').title(), color=color, alpha=0.85)
    ax2.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Loss Reduction from Initial', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Plot 3: Convergence Rate (loss delta per round)
    ax3 = axes[1, 0]
    for dept, color in zip(departments.keys(), colors):
        loss = departments[dept]["loss_per_round"]
        deltas = [abs(loss[i] - loss[i-1]) for i in range(1, len(loss))]
        ax3.plot(rounds[1:], deltas, marker='D', linewidth=2, markersize=6,
                label=dept.replace('_', ' ').title(), color=color, alpha=0.85)
    ax3.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax3.set_ylabel('|Δ Loss|', fontsize=11, fontweight='bold')
    ax3.set_title('Convergence Rate (Loss Change per Round)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    # Plot 4: Summary Statistics
    ax4 = axes[1, 1]
    dept_names = [d.replace('_', ' ').title() for d in departments.keys()]
    initial_loss = [departments[d]["loss_per_round"][0] for d in departments]
    final_loss = [departments[d]["loss_per_round"][-1] for d in departments]
    
    x = np.arange(len(dept_names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, initial_loss, width, label='Initial (Round 1)', color='#E74C3C', alpha=0.8)
    bars2 = ax4.bar(x + width/2, final_loss, width, label='Final (Round 10)', color='#27AE60', alpha=0.8)
    
    ax4.set_xlabel('Department', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax4.set_title('Initial vs Final Loss by Department', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(dept_names, rotation=15, ha='right', fontsize=9)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    combined_path = output_dir / "training_metrics_combined.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved combined metrics plot to: {combined_path}")


def create_summary_table():
    """Print a summary table of training results."""
    print("\n" + "="*70)
    print("FEDERATED LEARNING TRAINING SUMMARY")
    print("="*70)
    print(f"{'Department':<20} {'Initial Loss':<15} {'Final Loss':<15} {'Reduction':<15}")
    print("-"*70)
    
    for dept in training_data["departments"]:
        loss_data = training_data["departments"][dept]["loss_per_round"]
        initial = loss_data[0]
        final = loss_data[-1]
        reduction = (initial - final) / initial * 100
        
        dept_name = dept.replace('_', ' ').title()
        print(f"{dept_name:<20} {initial:<15.4f} {final:<15.4f} {reduction:<15.2f}%")
    
    # Overall average
    all_initial = [training_data["departments"][d]["loss_per_round"][0] for d in training_data["departments"]]
    all_final = [training_data["departments"][d]["loss_per_round"][-1] for d in training_data["departments"]]
    avg_initial = np.mean(all_initial)
    avg_final = np.mean(all_final)
    avg_reduction = (avg_initial - avg_final) / avg_initial * 100
    
    print("-"*70)
    print(f"{'AVERAGE':<20} {avg_initial:<15.4f} {avg_final:<15.4f} {avg_reduction:<15.2f}%")
    print("="*70)


if __name__ == "__main__":
    print("Generating training loss plots...")
    plot_training_loss()
    create_summary_table()
    print("\n✓ All plots generated successfully!")
    print("  - results/plots/training_loss_vs_rounds.png")
    print("  - results/plots/training_metrics_combined.png")
