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

# Aggregate data
df_agg = df.groupby(['dept', 'round_num'])['accuracy'].agg(['mean', 'std']).reset_index()

# Fill gaps by interpolating
df_agg = df_agg.set_index(['dept', 'round_num']).reindex(
    pd.MultiIndex.from_product([df['dept'].unique(), range(1,11)], names=['dept', 'round_num'])
).interpolate(method='linear').reset_index()

# Plot
plt.figure(figsize=(12, 8))
sns.lineplot(data=df_agg, x='round_num', y='mean', hue='dept', 
             errorbar='sd', marker='o', linewidth=4, markersize=10, markeredgewidth=1, estimator=None, sort=False)

# Add base lines
palette = sns.color_palette()
dept_list = list(base_accuracies.keys())
for i, (dept, base) in enumerate(base_accuracies.items()):
    plt.axhline(y=base, linestyle='--', color=palette[i % len(palette)], 
                label=f'{dept} base')

plt.xlabel('Round number (1-10)')
plt.ylabel('Mean accuracy (0-1)')
plt.title('Department-wise Convergence Plot')
plt.ylim(0, 1)
plt.xticks(range(1, 11)) 
plt.grid(True)

# Bottom legend
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=4, fontsize=11)
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
