# Results Directory

This directory contains all evaluation results, benchmarks, and training outputs from the DML project experiments.

## 📊 Evaluation Results

### Department-Specific Performance

#### `dept_specific_compare.json`
- **Purpose**: Compares base model vs department-specific LoRA adapters on custom test datasets
- **Datasets**: 200 samples per department (Finance, HR, Engineering, IT Support)
- **Metrics**: Accuracy scores and delta improvements over base model
- **Models**: Local department LoRA adapters (non-federated)

#### `federated_eval.json`
- **Purpose**: Evaluates federated learning results on department-specific tasks
- **Datasets**: Same 200 samples per department
- **Models**: Federated-trained adapters from `results/adapters/`
- **Comparison**: Shows federated vs local training performance

### GLUE Benchmark Results

#### `glue_comparison_general.json`
- **Purpose**: GLUE benchmark comparison (500 samples)
- **Tasks**: SST-2 (sentiment), MRPC (paraphrase), QQP, MNLI, QNLI, RTE, WNLI
- **Models**: Base model vs all department LoRA adapters
- **Use**: General language understanding evaluation

#### `glue_lora_base.json`
- **Purpose**: GLUE benchmark with smaller sample size (20 samples)
- **Same tasks** as above but with limited data for quick evaluation

#### `glue_adapted.json`, `glue_base_fixed.json`, `glue_client_results.json`
- **Purpose**: Additional GLUE evaluations with different configurations
- **glue_client_results.json**: GLUE performance of individual client adapters

### Perplexity Analysis

#### `perplexity_run.json`
- **Purpose**: Perplexity evaluation on department training datasets
- **Metrics**: Cross-domain perplexity (how well each model performs on other departments' data)
- **Models**: Base model vs LoRA adapters
- **Insight**: Measures domain specialization and generalization

#### `perplexity_benchmark_results.json`
- **Purpose**: Extended perplexity analysis with more models
- **Includes**: Multiple model variants and configurations

#### `residual_perplexity_results.json`
- **Purpose**: Perplexity for residual-based LoRA models
- **Models**: Experimental residual architectures

### Client-Level Results

#### `client_adapter_metrics.json`
- **Purpose**: Individual client adapter performance metrics
- **Scope**: All clients across all departments and rounds
- **Use**: Analyze client drift and individual performance

#### `client_eval_all_rounds.json`
- **Purpose**: Client performance across all federated rounds
- **Tracking**: Performance evolution during federated training
- **Rounds**: Round-by-round client accuracy

#### `client_eval_all_round5.json`
- **Purpose**: Client evaluation specifically for round 5
- **Snapshot**: Mid-training performance analysis

## 📈 Visualizations

### Generated Plots
- **`perplexity_heatmap.png`**: Heatmap visualization of perplexity results
- **`perplexity_accuracy.png`**: Accuracy plots for perplexity analysis

### Export Scripts
- **`export_results_for_plotting.py`**: Utility to export results in CSV/Python dict format for plotting

## 🏗️ Training Outputs

### Federated Training Logs
#### `client_runs/` directory
- **Structure**: `client_runs/{department}/{client_id}/`
- **Contents**: Training logs, metrics, and intermediate results per client
- **Note**: Currently empty - populated during federated training

### Residual Training
#### `residual_train/` directory
- **Contents**:
  - `qwen2-0.5b-instruction-residuals.json`: Residual data for Qwen2-0.5B
  - `qwen2.5-0.5b-instruction-residuals.json`: Residual data for Qwen2.5-0.5B
  - Various perplexity result files for residual models

### Synthetic Data Training
#### `synthetic_data_lora_train/` directory
- **Purpose**: Training outputs from synthetic data experiments
- **Contents**: LoRA adapters trained on synthetic datasets

## 🔍 Key Insights from Results

### Performance Summary
- **Base Model Accuracy**: ~0.93 on department-specific tasks
- **LoRA Improvement**: +0.5-1% accuracy gain per department
- **Federated vs Local**: Comparable performance with privacy benefits
- **Domain Specialization**: Models show best performance on their target domain

### GLUE Benchmarks
- **Base Model**: Moderate performance on general tasks
- **Department Models**: Mixed results - some domains help certain tasks
- **Best Performance**: Varies by department and GLUE task

### Perplexity Analysis
- **Cross-Domain**: Models generalize reasonably to other departments
- **Specialization**: Best perplexity on target domain data
- **Residual Models**: Experimental architectures show promise

## 📋 File Format Reference

### JSON Result Structure
```json
{
  "base_model": "Qwen/Qwen1.5-1.8B-Chat",
  "results": [
    {
      "department": "finance",
      "dataset": "app/dataset/test/FINANCE_test.jsonl",
      "num_samples": 200,
      "base": {"accuracy": 0.93},
      "models": [
        {
          "accuracy": 0.94,
          "id": "model_identifier",
          "delta_vs_base": 0.01
        }
      ]
    }
  ]
}
```

### GLUE Result Structure
```json
{
  "base": {
    "sst2": {"accuracy": 0.482, "f1": 0.044, "num_samples": 500}
  },
  "finance": {
    "sst2": {"accuracy": 0.485, "f1": 0.048, "num_samples": 500}
  }
}
```

## 🛠️ Usage

### For Analysis
```bash
# Export results for plotting
python export_results_for_plotting.py

# View specific results
cat dept_specific_compare.json | jq '.results[0]'
```

### For Model Selection
- Use `federated_eval.json` to compare federated vs local performance
- Use `glue_comparison_general.json` for general capability assessment
- Use `perplexity_run.json` to understand domain specialization

### For Research
- `client_adapter_metrics.json`: Study client drift in federated learning
- `residual_train/`: Analyze residual-based training approaches
- GLUE files: Benchmark against general language tasks

