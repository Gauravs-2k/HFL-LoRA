import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the client-specific evaluation results
with open('results/client_specific_eval.json', 'r') as f:
    data_json = json.load(f)

# Load base accuracies
with open('results/base_eval_all_rounds.json', 'r') as f:
    base_json = json.load(f)
base_accuracies = base_json['base_accuracies']

# Prepare data for DataFrame
data = []
client_results = data_json['client_results']
for dept_name, rounds in client_results.items():
    for round_key, clients in rounds.items():
        round_num = int(round_key.split('_')[1])
        for client in clients:
            accuracy = client['accuracy']
            data.append({
                'dept': dept_name,
                'round_num': round_num,
                'accuracy': accuracy
            })

df = pd.DataFrame(data)

# Determine if multi-round
multi_round = df['round_num'].nunique() > 1
hue = 'round_num' if multi_round else None

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='dept', y='accuracy', hue=hue,
            linewidth=2.5,            # Thick box lines
            flierprops=dict(marker='', markersize=0),  # Hide outliers
            medianprops=dict(color='black', linewidth=3),  # Fat median
            whiskerprops=dict(linewidth=2),  # Thick whiskers
            capprops=dict(linewidth=2),      # Cap ends
            boxprops=dict(facecolor='lightblue', alpha=0.7))
if multi_round:
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Round')  # Force if hue

# Add base horizontal lines
palette = sns.color_palette()
for i, dept in enumerate(df['dept'].unique()):
    plt.axhline(base_accuracies[dept], ls='--', color=palette[i % len(palette)], alpha=0.8,
                label=f'{dept} base')

plt.legend()

# Polish
sns.despine()
plt.grid(axis='y', alpha=0.3)
plt.ylim(0, 1.05)

plt.xlabel('Department')
plt.ylabel('Client accuracy')
plt.title('Client Accuracy Distribution by Department')
plt.savefig('distribution_boxplot.pdf')
plt.show()
