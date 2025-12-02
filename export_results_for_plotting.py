import json

def export_results():
    # Load all evaluation results
    with open('results/dept_specific_compare.json') as f:
        local_data = json.load(f)
    
    with open('results/federated_eval.json') as f:
        fed_data = json.load(f)
    
    print("EVALUATION RESULTS FOR PLOTTING")
    print("=" * 50)
    print()
    
    # CSV format for easy plotting
    print("CSV FORMAT:")
    print("Department,Model_Type,Base_Accuracy,Adapter_Accuracy,Delta_vs_Base")
    
    departments = ['finance', 'hr', 'engineering', 'it_support']
    
    for dept in departments:
        # Local results
        local_result = next(r for r in local_data['results'] if r['department'] == dept)
        local_base = local_result['base']['accuracy']
        local_adapter = local_result['models'][0]['accuracy']
        local_delta = local_result['models'][0]['delta_vs_base']
        
        print(f"{dept},Local,{local_base:.4f},{local_adapter:.4f},{local_delta:.4f}")
        
        # Federated results  
        fed_result = next(r for r in fed_data['results'] if r['department'] == dept)
        fed_base = fed_result['base']['accuracy']
        fed_adapter = fed_result['models'][0]['accuracy']
        fed_delta = fed_result['models'][0]['delta_vs_base']
        
        print(f"{dept},Federated,{fed_base:.4f},{fed_adapter:.4f},{fed_delta:.4f}")
    
    print()
    print("PYTHON DICT FORMAT:")
    print("results = {")
    print("    'departments': ['finance', 'hr', 'engineering', 'it_support'],")
    print("    'local': {")
    
    for dept in departments:
        local_result = next(r for r in local_data['results'] if r['department'] == dept)
        print(f"        '{dept}': {{")
        print(f"            'base_accuracy': {local_result['base']['accuracy']},")
        print(f"            'adapter_accuracy': {local_result['models'][0]['accuracy']},")
        print(f"            'delta_vs_base': {local_result['models'][0]['delta_vs_base']}")
        print("        },")
    
    print("    },")
    print("    'federated': {")
    
    for dept in departments:
        fed_result = next(r for r in fed_data['results'] if r['department'] == dept)
        print(f"        '{dept}': {{")
        print(f"            'base_accuracy': {fed_result['base']['accuracy']},")
        print(f"            'adapter_accuracy': {fed_result['models'][0]['accuracy']},")
        print(f"            'delta_vs_base': {fed_result['models'][0]['delta_vs_base']}")
        print("        },")
    
    print("    }")
    print("}")

if __name__ == "__main__":
    export_results()
