import json

def compare_rounds(round1_file, round5_file):
    with open(round1_file) as f:
        round1 = json.load(f)
    with open(round5_file) as f:
        round5 = json.load(f)
    
    print('=' * 80)
    print('FEDERATED LEARNING PROGRESS: ROUND 1 vs ROUND 5')
    print('=' * 80)
    print(f'{"Department":<15} {"Round 1":>10} {"Round 5":>10} {"Improvement":>12} {"vs Local":>10}')
    print('-' * 80)
    
    # Load local baseline for comparison
    with open('results/dept_specific_compare.json') as f:
        local = json.load(f)
    local_by_dept = {r['department']: r for r in local['results']}
    
    r1_by_dept = {r['department']: r for r in round1['results']}
    r5_by_dept = {r['department']: r for r in round5['results']}
    
    for dept in ['finance', 'hr', 'engineering', 'it_support']:
        r1_acc = r1_by_dept.get(dept, {}).get('models', [{}])[0].get('accuracy', 0)
        r5_acc = r5_by_dept.get(dept, {}).get('models', [{}])[0].get('accuracy', 0)
        local_acc = local_by_dept.get(dept, {}).get('models', [{}])[0].get('accuracy', 0)
        
        improvement = r5_acc - r1_acc
        vs_local = r5_acc - local_acc
        
        print(f'{dept:<15} {r1_acc:>10.1%} {r5_acc:>10.1%} {improvement:>+12.1%} {vs_local:>+10.1%}')
    
    print('=' * 80)

if __name__ == "__main__":
    compare_rounds('results/federated_eval.json', 'results/federated_eval_round5.json')
