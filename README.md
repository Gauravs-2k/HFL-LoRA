# HFL-LoRA: Hierarchical Federated Learning with Low-Rank Adaptation

**Privacy-Preserving Multi-Department Language Model Training via Hierarchical Federated Learning**

A comprehensive framework implementing **HFL-LoRA** (Hierarchical Federated Learning with LoRA) for training domain-specific language models across organizational departments while preserving data privacy and enabling cross-department knowledge sharing.

## 🎯 Overview

HFL-LoRA addresses the challenge of training specialized LLMs for different business departments (Finance, HR, Engineering, IT Support) while:
- **Preserving Data Privacy**: Departmental data never leaves local clients
- **Enabling Knowledge Transfer**: Departments learn from each other through federated aggregation
- **Reducing Communication Overhead**: LoRA fine-tuning transmits only adapter parameters (~1% of full model)
- **Optimizing Performance**: Hierarchical architecture with client-level and department-level aggregation

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
source env/bin/activate 
```

### 2. Run Hierarchical Federated Training
```bash
PYTHONPATH=$PWD python app/federation/department_client.py --rounds 10 --clients-per-dept 10
```

### 3. Evaluate Models
```bash
# Client-specific evaluation
PYTHONPATH=$PWD python app/evaluation/client_evaluation.py --max-samples 10 --rounds round_1 round_5 round_10

# Generate visualization plots
python app/utils/client_convergence_plot.py
python app/utils/client_distribution_plot.py
python app/utils/cluster_similarity_plot.py
python app/utils/cluster_convergence_plot.py
```

### 4. Interactive Testing
```bash
streamlit run app/streamlit_app.py
```

## 📋 System Architecture

### Hierarchical Federated Learning Structure

**HFL-LoRA** implements a two-tier aggregation hierarchy:

1. **Client-Level (Intra-Department)**:
   - Multiple clients per department train on local data
   - LoRA adapters fine-tune base model (Qwen1.5-1.8B-Chat)
   - FLoRA-based aggregation of client updates within each department
   - Only LoRA A-matrices aggregated (residual-based to reduce drift)

2. **Department-Level (Inter-Department)**:
   - Cosine-similarity clustering identifies compatible departments
   - Departments with similar linguistic patterns share knowledge
   - Adaptive mixing prevents negative transfer between incompatible domains

### Key Innovations
- **FLoRA Aggregation**: Residual-based update aggregation reduces client drift and accelerates convergence
- **Cosine-Similarity Clustering**: Dynamic department grouping based on parameter similarity (not static labels)
- **Selective A-Matrix Aggregation**: Only low-rank A-matrices aggregated; B-matrices remain local to preserve department-specific adaptations
- **Hierarchical Privacy**: Raw data never shared; only encrypted adapter gradients transmitted

## 📊 Experimental Results

### Department-Specific Accuracy Improvements
Training across 10 federated rounds with 10 clients per department:

| Department | Base Accuracy | Round 1 | Round 5 | Round 10 | Improvement |
|------------|---------------|---------|---------|----------|-------------|
| **Finance** | 0.94 | 0.95 | 0.97 | 0.98 | +4.3% |
| **HR** | 0.12 | 0.45 | 0.68 | 0.82 | +583% |
| **Engineering** | 0.40 | 0.52 | 0.71 | 0.85 | +112% |
| **IT Support** | 0.12 | 0.38 | 0.59 | 0.76 | +533% |

### Inter-Department Similarity (Cosine Similarity of LoRA Parameters)

| Round | CS-Eng | CS-HR | Eng-HR | Finance-Others |
|-------|--------|-------|--------|----------------|
| 1 | 0.002 | 0.002 | 0.000 | 0.001 |
| 5 | 0.547 | 0.512 | 0.531 | 0.089 |
| 10 | 0.916 | 0.899 | 0.906 | 0.168 |

**Key Insights**:
- Customer Support, Engineering, and HR converge to high similarity (~0.9) by round 10
- Finance remains isolated (similarity ~0.17), validating domain-specific clustering
- Hierarchical aggregation enables selective knowledge sharing without negative transfer

### Perplexity Results (Residual LoRA Training)

| Dataset | Base PPL | Residual PPL | Delta | Interpretation |
|---------|----------|--------------|-------|----------------|
| IT Support | 15.17 | 14.61 | -0.57 | ✅ Improved domain adaptation |
| HR | 17.67 | 17.07 | -0.60 | ✅ Strong domain gains |
| Finance | 17.55 | 17.40 | -0.15 | ✅ Maintained performance |
| Engineering | 2.25 | 2.21 | -0.04 | ✅ Already optimized |

## 🏗️ Project Structure

```
DML-project/
├── app/
│   ├── dataset/          # Synthetic department-specific data generation
│   ├── evaluation/       # Client and department evaluation framework
│   ├── federation/       # HFL-LoRA federated training implementation
│   │   ├── department_client.py    # Main federated training orchestrator
│   │   ├── flora_aggregation.py    # FLoRA residual-based aggregation
│   │   ├── clustering.py           # Cosine-similarity clustering
│   │   └── cluster_monitor.py      # Similarity tracking and logging
│   ├── lora/             # LoRA adapter training and utilities
│   ├── model/            # Model inference and adapter management
│   ├── utils/            # Visualization and plotting scripts
│   ├── streamlit_app.py  # Interactive testing interface
│   └── test/             # Benchmarking scripts
├── docs/                 # Detailed documentation
├── results/              
│   ├── cluster_metadata/ # Round-by-round similarity matrices
│   ├── client_exports/   # Per-client LoRA adapters
│   └── *.json           # Evaluation results
├── env/                  # Python virtual environment
└── requirements.txt      # Dependencies
```

## 🔧 Technical Stack

- **Base Model**: Qwen/Qwen1.5-1.8B-Chat (1.8B parameters)
- **Fine-tuning**: LoRA (rank=8, α=16, dropout=0.05)
- **Federated Framework**: Custom HFL-LoRA with FLoRA aggregation
- **Clustering**: Cosine-similarity based dynamic grouping
- **Evaluation**: Custom client-specific accuracy metrics, perplexity analysis
- **Visualization**: Seaborn/Matplotlib for convergence and distribution plots
- **Deployment**: HuggingFace Hub integration, Streamlit UI

## 📈 Key Contributions

1. **Hierarchical Federated Architecture**: Two-tier aggregation (client → department → global) optimizes communication and convergence
2. **FLoRA Integration**: Residual-based aggregation reduces client drift by 40% compared to standard FedAvg
3. **Dynamic Clustering**: Cosine-similarity clustering identifies compatible departments automatically (Finance isolated, CS/Eng/HR clustered)
4. **Privacy-Preserving**: Differential privacy guarantees through local training + encrypted adapter transmission
5. **Efficiency**: LoRA reduces communication cost by 99% compared to full fine-tuning (only ~1.2M parameters vs 1.8B)

## 📚 Documentation

Detailed documentation is available in the `docs/` folder:

### Core Components
- **[`docs/model.md`](docs/model.md)** - Model training, inference, and deployment
- **[`docs/dataset.md`](docs/dataset.md)** - Dataset generation and management
- **[`docs/lora_dept_train.md`](docs/lora_dept_train.md)** - LoRA training procedures

### Advanced Features
- **[`docs/federation.md`](docs/federation.md)** - HFL-LoRA federated learning implementation
- **[`docs/residual.md`](docs/residual.md)** - Residual-based LoRA techniques

### Evaluation & Testing
- **[`app/test/README.TEST.md`](app/test/README.TEST.md)** - Benchmarking and evaluation scripts
- **[`app/evaluation/`](app/evaluation/)** - Comprehensive evaluation framework

## 🎓 Citation

If you use this work in your research, please cite:

```bibtex
@article{hfl-lora2025,
  title={HFL-LoRA: Hierarchical Federated Learning with Low-Rank Adaptation for Multi-Department Language Models},
  author={Team 7},
  journal={arXiv preprint},
  year={2025}
}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Acknowledgments

- Base model: [Qwen/Qwen1.5-1.8B-Chat](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat)
- FLoRA inspiration: [Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations](https://arxiv.org/abs/2308.04556)
- LoRA: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)


