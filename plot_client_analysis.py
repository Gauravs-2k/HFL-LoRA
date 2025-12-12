"""
Client-Level Federated Learning Analysis Plots

Visualizes individual client performance across departments and rounds.
Shows distribution of client accuracies, performance over training rounds,
and comparison with department-level aggregation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_client_results(results_file: str = "results/client_adapter_metrics.json") -> Dict[str, Any]:
    """Load client evaluation results."""
    path = Path(results_file)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(path, 'r') as f:
        return json.load(f)

def plot_client_performance_distribution(results: Dict[str, Any], output_dir: Path = Path("results/plots")):
    """Plot distribution of client accuracies within each department."""
    print("Generating client performance distribution plot...")

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    departments = ['engineering', 'finance', 'hr', 'customer_support']

    for idx, dept in enumerate(departments):
        ax = axes[idx]

        if dept not in results.get('client_results', {}):
            ax.text(0.5, 0.5, f'No data for {dept}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{dept.replace("_", " ").title()}')
            continue

        dept_data = results['client_results'][dept]

        # Collect all client accuracies across rounds
        all_accuracies = []
        round_labels = []

        for round_name, clients in dept_data.items():
            round_num = int(round_name.split('_')[1])
            client_accs = [c.get('accuracy', 0) for c in clients if 'accuracy' in c]
            if client_accs:
                all_accuracies.extend(client_accs)
                round_labels.extend([f'R{round_num}'] * len(client_accs))

        if all_accuracies:
            # Create violin plot with individual points
            sns.violinplot(data=all_accuracies, ax=ax, inner='quartile', color='lightblue')
            sns.stripplot(data=all_accuracies, ax=ax, size=4, alpha=0.6, color='darkblue')

            ax.set_title(f'{dept.replace("_", " ").title()} Client Performance')
            ax.set_ylabel('Accuracy')
            ax.grid(True, alpha=0.3)

            # Add statistics
            mean_acc = np.mean(all_accuracies)
            std_acc = np.std(all_accuracies)
            ax.axhline(y=mean_acc, color='red', linestyle='--', alpha=0.7, label='.2f')
            ax.legend()

        else:
            ax.text(0.5, 0.5, f'No client data for {dept}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{dept.replace("_", " ").title()}')

    plt.tight_layout()
    plt.savefig(output_dir / 'client_performance_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'client_performance_distribution.png'}")

def plot_client_performance_over_rounds(results: Dict[str, Any], output_dir: Path = Path("results/plots")):
    """Plot average client performance progression over training rounds."""
    print("Generating client performance over rounds plot...")

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    departments = ['engineering', 'finance', 'hr', 'customer_support']

    for idx, dept in enumerate(departments):
        ax = axes[idx]

        if dept not in results.get('client_results', {}):
            ax.text(0.5, 0.5, f'No data for {dept}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{dept.replace("_", " ").title()}')
            continue

        dept_data = results['client_results'][dept]

        # Calculate average client accuracy per round
        rounds = []
        avg_accuracies = []
        client_counts = []

        for round_name, clients in sorted(dept_data.items()):
            round_num = int(round_name.split('_')[1])
            client_accs = [c.get('accuracy', 0) for c in clients if 'accuracy' in c]

            if client_accs:
                rounds.append(round_num)
                avg_accuracies.append(np.mean(client_accs))
                client_counts.append(len(client_accs))

        if rounds:
            # Plot with client count as marker size
            scatter = ax.scatter(rounds, avg_accuracies, s=[c*20 for c in client_counts],
                               alpha=0.7, color='darkblue', edgecolors='black')

            # Add trend line
            if len(rounds) > 1:
                z = np.polyfit(rounds, avg_accuracies, 2)
                p = np.poly1d(z)
                x_trend = np.linspace(min(rounds), max(rounds), 100)
                ax.plot(x_trend, p(x_trend), '--', color='red', alpha=0.7, label='Trend')

            ax.set_title(f'{dept.replace("_", " ").title()} - Avg Client Performance')
            ax.set_xlabel('Training Round')
            ax.set_ylabel('Average Client Accuracy')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add client count annotation
            for r, acc, count in zip(rounds, avg_accuracies, client_counts):
                ax.annotate(f'n={count}', (r, acc), xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)

        else:
            ax.text(0.5, 0.5, f'No round data for {dept}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{dept.replace("_", " ").title()}')

    plt.tight_layout()
    plt.savefig(output_dir / 'client_performance_over_rounds.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'client_performance_over_rounds.png'}")

def plot_client_vs_department_comparison(results: Dict[str, Any], output_dir: Path = Path("results/plots")):
    """Compare individual client performance vs department aggregated performance."""
    print("Generating client vs department comparison plot...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load department-level results if available
    dept_results_file = Path("results/glue_client_results.json")
    dept_accuracies = {}

    if dept_results_file.exists():
        try:
            with open(dept_results_file, 'r') as f:
                dept_data = json.load(f)
                for dept, metrics in dept_data.items():
                    if 'accuracy' in metrics:
                        dept_accuracies[dept] = metrics['accuracy']
        except:
            pass

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    departments = ['engineering', 'finance', 'hr', 'customer_support']

    for idx, dept in enumerate(departments):
        ax = axes[idx]

        if dept not in results.get('client_results', {}):
            ax.text(0.5, 0.5, f'No data for {dept}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{dept.replace("_", " ").title()}')
            continue

        dept_data = results['client_results'][dept]

        # Get final round client accuracies
        final_round = max([int(r.split('_')[1]) for r in dept_data.keys()])
        final_round_name = f'round_{final_round}'

        if final_round_name in dept_data:
            clients = dept_data[final_round_name]
            client_accs = [c.get('accuracy', 0) for c in clients if 'accuracy' in c]

            if client_accs:
                # Plot client accuracies as scatter
                y_pos = [1] * len(client_accs)
                ax.scatter(client_accs, y_pos, alpha=0.7, s=50, color='blue', label='Individual Clients')

                # Add department aggregated accuracy if available
                if dept in dept_accuracies:
                    ax.axvline(x=dept_accuracies[dept], color='red', linestyle='--',
                              linewidth=2, label='.4f')

                # Add base model accuracy
                base_acc = results.get('base_accuracies', {}).get(dept, 0)
                if base_acc > 0:
                    ax.axvline(x=base_acc, color='green', linestyle=':', linewidth=2, label='.4f')

                ax.set_title(f'{dept.replace("_", " ").title()} - Final Round (R{final_round})')
                ax.set_xlabel('Accuracy')
                ax.set_yticks([])
                ax.grid(True, alpha=0.3)
                ax.legend()

                # Add statistics
                mean_client = np.mean(client_accs)
                std_client = np.std(client_accs)
                ax.text(0.02, 0.98, f'Clients: μ={mean_client:.3f}, σ={std_client:.3f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'client_vs_department_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'client_vs_department_comparison.png'}")

def generate_all_client_plots(results_file: str = "results/client_adapter_metrics.json"):
    """Generate all client-level analysis plots."""
    print("Generating comprehensive client-level analysis plots...")
    print("=" * 60)

    try:
        results = load_client_results(results_file)
        output_dir = Path("results/plots")

        plot_client_performance_distribution(results, output_dir)
        plot_client_performance_over_rounds(results, output_dir)
        plot_client_vs_department_comparison(results, output_dir)

        print("\n" + "=" * 60)
        print("✓ All client analysis plots generated successfully!")
        print(f"✓ Plots saved to: {output_dir}/")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        return False

    return True

if __name__ == "__main__":
    generate_all_client_plots()