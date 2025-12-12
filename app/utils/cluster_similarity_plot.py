import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

rounds_to_plot = [1, 5, 10]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, round_num in enumerate(rounds_to_plot):
    with open(f'results/cluster_metadata/round_{round_num}_similarity.json', 'r') as f:
        data = json.load(f)
    
    similarity_matrix = np.array(data['similarity_matrix'])
    dept_names = data['names']
    
    ax = axes[idx]
    sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='YlOrRd', 
                xticklabels=dept_names, yticklabels=dept_names,
                vmin=0, vmax=1, cbar_kws={'label': 'Cosine Similarity'},
                linewidths=0.5, ax=ax)
    ax.set_title(f'Round {round_num}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Department', fontsize=11)
    ax.set_ylabel('Department', fontsize=11)

plt.suptitle('Inter-Department Similarity Evolution', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('cluster_similarity_heatmap.pdf', bbox_inches='tight')
plt.show()
