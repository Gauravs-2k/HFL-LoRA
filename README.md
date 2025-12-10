# DML-Project: Federated LoRA Fine-tuning for Department-Specific LLMs

A comprehensive framework for **federated learning** with **LoRA adapters** to create specialized language models for different business departments (Finance, HR, Engineering, IT Support).

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
source env/bin/activate 
```

### 2. Train Department Models
```bash
python app/lora/lora.py
```

### 3. Test Models Interactively
```bash
streamlit run app/streamlit_app.py
```

### 4. Run Federated Training (Optional)
```bash
PYTHONPATH=$PWD python app/federation/department_client.py --rounds 10 --clients-per-dept 3
```

## 📋 Project Overview

This project implements **FedSA-LoRA** (Federated Learning with Shared A-matrices LoRA) using **FLoRA** (residual-based aggregation) for training department-specific language models in a privacy-preserving manner.

### Key Features
- **Department-Specific Models**: Specialized LLMs for Finance, HR, Engineering, IT Support
- **Federated Learning**: Privacy-preserving distributed training
- **LoRA Fine-tuning**: Efficient parameter-efficient training
- **Interactive Testing**: Streamlit UI for model evaluation
- **Comprehensive Evaluation**: GLUE benchmarks, perplexity analysis, accuracy metrics

## 📚 Documentation

Detailed documentation is available in the `docs/` folder:

### Core Components
- **[`docs/model.md`](docs/model.md)** - Model training, inference, and deployment
- **[`docs/dataset.md`](docs/dataset.md)** - Dataset generation and management
- **[`docs/lora_dept_train.md`](docs/lora_dept_train.md)** - LoRA training procedures

### Advanced Features
- **[`docs/federation.md`](docs/federation.md)** - Federated learning implementation and configuration
- **[`docs/residual.md`](docs/residual.md)** - Residual-based LoRA techniques

### Evaluation & Testing
- **[`app/test/README.TEST.md`](app/test/README.TEST.md)** - Benchmarking and evaluation scripts
- **[`app/evaluation/`](app/evaluation/)** - Comprehensive evaluation framework

## 🏗️ Project Structure

```
DML-project/
├── app/
│   ├── dataset/          # Data generation and management
│   ├── evaluation/       # Model evaluation framework
│   ├── federation/       # Federated learning implementation
│   ├── lora/            # LoRA training scripts
│   ├── model/           # Model inference and utilities
│   ├── streamlit_app.py # Interactive testing interface
│   └── test/            # Benchmarking scripts
├── docs/                # Detailed documentation
├── results/             # Training outputs and evaluations
├── env/                 # Python virtual environment
└── requirements.txt     # Dependencies
```

## 🔧 Key Technologies

- **Base Model**: Qwen/Qwen1.5-1.8B-Chat
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Federated Learning**: Flower framework with custom FLoRA aggregation
- **Evaluation**: GLUE benchmarks, perplexity analysis
- **Deployment**: HuggingFace Hub, Docker, REST API
- **UI**: Streamlit for interactive testing



