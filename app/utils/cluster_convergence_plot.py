import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = []
for round_num in range(1, 11):
    with open(f'results/cluster_metadata/round_{round_num}_similarity.json', 'r') as f:
        sim_data = json.load(f)
    
    similarity_matrix = sim_data['similarity_matrix']
    dept_names = sim_data['names']
    
    for i in range(len(dept_names)):
        for j in range(i+1, len(dept_names)):
            data.append({
                'round': round_num,
                'pair': f'{dept_names[i]}-{dept_names[j]}',
                'similarity': similarity_matrix[i][j]
            })

df = pd.DataFrame(data)

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='round', y='similarity', hue='pair', 
             marker='o', linewidth=3, markersize=8)

plt.xlabel('Round Number', fontsize=12)
plt.ylabel('Cosine Similarity', fontsize=12)
plt.title('Pairwise Department Similarity Convergence', fontsize=14, fontweight='bold')
plt.ylim(-0.05, 1.05)
plt.xticks(range(1, 11))
plt.grid(axis='y', alpha=0.3)
plt.legend(title='Department Pairs', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('cluster_convergence_lineplot.pdf', bbox_inches='tight')
plt.show()
