# Federated Learning System

This directory contains the core federated learning implementation using **FedSA-LoRA** (Federated Learning with Shared A-matrices LoRA) and **FLoRA** (residual-based aggregation) with hierarchical clustering.

## Architecture Overview

### FedSA-LoRA + FLoRA + Clustering

The system implements a **hierarchical federated learning** approach:

```
Client Level → Department Level → Cross-Department Clustering
     ↓              ↓                    ↓
   LoRA           FLoRA              K-means
 Training      Aggregation          Clustering
```

**Key Innovations:**
- **FedSA-LoRA**: Shares only LoRA A-matrices for clustering decisions
- **FLoRA**: Residual-based aggregation to reduce client drift
- **Hierarchical Clustering**: Department-level grouping using k-means + elbow method

## Directory Structure

```
app/federation/
├── readme.md                           # This file
├── cluster_config.py                   # Configuration constants
├── cluster_monitor.py                  # Logging and monitoring utilities
├── clustering.py                       # K-means clustering algorithms
├── department_client.py                # Main federated training simulation
├── flora_aggregation.py                # FLoRA residual aggregation
├── lora_utils.py                       # LoRA model utilities
└── __pycache__/                       # Python cache files
```

## Core Components

### Main Entry Point

#### `department_client.py`
**Purpose**: Orchestrates the entire federated learning simulation

**Key Functions:**
- `simulate_sequential_training()`: Main training loop
- `DepartmentLoraClient`: Flower-compatible client class

**Architecture:**
```python
# Simulation mode (not distributed)
for round in 1 to rounds:
    for department in departments:
        # Train clients within department
        # FLoRA aggregate client updates
        # Export client adapters

    # Cluster departments by LoRA similarity
    # Cross-department aggregation
    # Update department models
```

### Aggregation Methods

#### `flora_aggregation.py`
**Purpose**: Implements FLoRA (Federated Fine-Tuning with Residual Aggregation)

**Key Features:**
- **Residual-based aggregation**: Prevents client drift
- **Momentum updates**: Stabilizes convergence
- **Weighted averaging**: Handles heterogeneous client data

**Core Function:**
```python
def flora_weighted_aggregate(
    items: List[Tuple[parameters, weight]],
    previous_global: parameters,
    momentum: float = 0.1,
    use_residual: bool = True
) -> parameters
```

### Clustering System

#### `clustering.py`
**Purpose**: Department-level clustering using LoRA parameter similarity

**Algorithms:**
- **K-means++ initialization**: Smart centroid selection
- **Lloyd's algorithm**: Iterative clustering
- **Elbow method**: Automatic optimal cluster detection

**Key Functions:**
- `cluster_aware_average_selected()`: Client-level clustering within departments
- `department_level_clustering()`: Cross-department clustering
- `determine_optimal_clusters()`: Elbow method for k-selection

#### `cluster_monitor.py`
**Purpose**: Logging and monitoring of clustering decisions

**Features:**
- Cluster assignment logging
- Similarity matrix computation
- Aggregation statistics tracking
- Metadata export for analysis

### Configuration

#### `cluster_config.py`
**Purpose**: Centralized configuration for all clustering and aggregation parameters

**Key Settings:**
```python
# Elbow method
ELBOW_MAX_CLUSTERS = 5
ELBOW_MIN_CLUSTERS = 2

# FLoRA parameters
FLORA_MOMENTUM = 0.1
FLORA_USE_RESIDUAL = True

# Department clustering
DEPT_CLUSTERING_ENABLED = True
DEPT_MAX_CLUSTERS = 5
DEPT_CLUSTER_MIXING = 0.3

# Logging
ENABLE_CLUSTER_LOGGING = True
SAVE_CLUSTER_METADATA = True
```

### Utilities

#### `lora_utils.py`
**Purpose**: LoRA model creation, parameter handling, and adapter management

**Key Functions:**
- `create_lora_model()`: Initialize LoRA models
- `collect_lora_state()`: Extract LoRA parameters
- `export_lora_adapter()`: Save trained adapters
- `load_adapter_model()`: Load existing adapters

## Federated Learning Flow

### Phase 1: Client Training Within Departments

```python
for department in departments:
    for client in department_clients:
        # Load department model state
        # Train on local data
        # Export client adapter
        # Send LoRA A-matrices to department aggregation
```

### Phase 2: Intra-Department FLoRA Aggregation

```python
# Aggregate client updates using FLoRA
department_state = flora_weighted_aggregate(
    client_states,  # Only A-matrices
    previous_dept_state,
    momentum=0.1
)
```

### Phase 3: Department-Level Clustering

```python
# Extract FLoRA-aggregated A-matrices
dept_a_matrices = [dept_state[a_indices] for dept_state in department_states]

# Cluster departments using k-means + elbow method
clusters, cluster_assignments = department_level_clustering(
    dept_a_matrices,
    method="elbow"
)
```

### Phase 4: Cross-Department Aggregation

```python
# Aggregate within clusters
for cluster_id, dept_list in clusters.items():
    cluster_state = flora_weighted_aggregate(
        [department_states[dept] for dept in dept_list]
    )

# Mix department states with cluster averages
for dept in departments:
    cluster_id = cluster_assignments[dept]
    dept_state = mix_with_cluster_average(
        department_states[dept],
        cluster_states[cluster_id],
        mixing_ratio=0.3
    )
```

## Configuration Options

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rounds` | 10 | Number of federated rounds |
| `clients-per-dept` | 3 | Clients per department |
| `local-epochs` | 1 | Training epochs per client |
| `learning-rate` | 0.0002 | Client learning rate |
| `max-records` | 128 | Training samples per client |

### LoRA Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora-r` | 8 | LoRA rank |
| `lora-alpha` | 16 | LoRA alpha |
| `lora-dropout` | 0.05 | LoRA dropout |
| `target-modules` | `["q_proj", "v_proj"]` | Target attention modules |

### Clustering Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num-clusters` | 2 | Client clusters within departments |
| `global-mix` | 0.2 | Global mixing ratio |
| `dept-clustering` | enabled | Department-level clustering |

## Usage Examples

### Basic Training Run

```bash
# Run 10 rounds with 3 clients per department
source env/bin/activate
PYTHONPATH=$PWD python app/federation/department_client.py \
  --rounds 10 \
  --clients-per-dept 3 \
  --global-mix 0.2
```

### Custom Configuration

```bash
# Higher learning rate, more local epochs
python app/federation/department_client.py \
  --rounds 5 \
  --clients-per-dept 5 \
  --learning-rate 0.0005 \
  --local-epochs 2 \
  --global-mix 0.3
```

### Memory-Efficient Training

```bash
# Use 4-bit quantization
python app/federation/department_client.py \
  --rounds 10 \
  --load-in-4bit \
  --device-map "auto"
```

## Output Structure

### Results Directory Structure

```
results/
├── adapters/                          # Department-level adapters
│   ├── engineering/
│   ├── finance/
│   ├── hr/
│   └── customer_support/
├── client_exports/                    # Client adapters by round
│   ├── engineering/
│   │   ├── round_1_client_0/
│   │   ├── round_1_client_1/
│   │   └── ...
│   └── ...
├── cluster_metadata/                  # Clustering logs
│   ├── round_1_department_clusters.json
│   ├── round_1_similarity.json
│   └── ...
└── client_runs/                       # Training logs
    ├── engineering/
    ├── finance/
    └── ...
```

### Key Output Files

- **`client_lora_federated.json`**: Maps all client adapters by round
- **`dept_lora_federated.json`**: Department adapter paths
- **`federated_eval.json`**: Evaluation results
- **`client_eval_all_rounds.json`**: Comprehensive client performance

## Performance Characteristics

### Memory Usage
- **Base Model**: Qwen1.5-1.8B (~3.5GB)
- **LoRA Overhead**: ~50MB per adapter
- **GPU Memory**: 4-8GB during training

### Training Time
- **Per Round**: 5-15 minutes (depending on data size)
- **Client Training**: 1-3 minutes each
- **Clustering**: < 1 second
- **Total (10 rounds)**: 1-2 hours

### Scaling
- **Clients**: Linear scaling with client count
- **Departments**: Clustering overhead grows with O(n²)
- **Rounds**: Accumulates adapter storage

## Algorithm Details

### FLoRA Aggregation

**Residual Calculation:**
```python
residuals = current_params - previous_params
aggregated_residuals = weighted_average(residuals)
new_params = previous_params + momentum * aggregated_residuals
```

**Benefits:**
- Reduces client drift
- Stabilizes convergence
- Handles heterogeneous data distributions

### Clustering with Elbow Method

**Optimal K Detection:**
```python
for k in range(min_k, max_k+1):
    centroids, labels = kmeans(X, k)
    inertia = sum_squared_distances(X, centroids)

# Find elbow using second derivative
second_deriv = d²inertia/dk²
optimal_k = argmax(second_deriv) + min_k
```

**Benefits:**
- Automatic cluster count selection
- Adapts to data structure
- Prevents overfitting

### FedSA-LoRA Design

**Parameter Sharing Strategy:**
- **A-matrices**: Shared for clustering (capture attention patterns)
- **B-matrices**: Local adaptation (department-specific)
- **Clustering**: Based on A-matrix similarity
- **Aggregation**: All parameters via FLoRA

**Benefits:**
- Efficient clustering decisions
- Preserves local adaptation
- Reduces communication overhead

## Integration Points

### Data Pipeline
- **Input**: `app/dataset/*_personal_clients/` (client data)
- **Output**: `results/adapters/` (trained models)
- **Evaluation**: `client_evaluation.py` (performance testing)

### Model Management
- **Configuration**: `app/model/*.json` (adapter paths)
- **Serving**: `app/model/inference.py` (API server)
- **Deployment**: `app/model/upload_adapters.py` (HuggingFace)

### Monitoring
- **Logs**: `results/cluster_metadata/` (clustering decisions)
- **Metrics**: `results/client_eval_all_rounds.json` (performance)
- **Visualization**: `create_comprehensive_plots.py` (analysis)

## Troubleshooting

### Common Issues

**Memory Errors:**
```bash
# Use 4-bit quantization
--load-in-4bit --device-map "auto"

# Reduce batch size
--batch-size 1
```

**Clustering Failures:**
```bash
# Check data availability
ls app/dataset/*_personal_clients/

# Verify LoRA parameters
python -c "from app.federation.lora_utils import collect_lora_parameter_names; print(len(collect_lora_parameter_names(model)))"
```

**Slow Training:**
```bash
# Use GPU acceleration
--device-map "cuda"

# Reduce precision
--dtype "fp16"
```

### Debugging

**Enable Verbose Logging:**
```python
# In cluster_config.py
ENABLE_CLUSTER_LOGGING = True
SAVE_CLUSTER_METADATA = True
```

**Check Cluster Assignments:**
```bash
# View clustering results
cat results/cluster_metadata/round_10_department_clusters.json
```

**Monitor Memory Usage:**
```python
# Add to training loop
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
```

## Future Extensions

### Distributed Mode
- Replace simulation with actual Flower server/client network
- Enable `server.py`, `laptop_client.py`, `mobile_client.py`

### Advanced Clustering
- Silhouette analysis for cluster validation
- Hierarchical clustering alternatives
- Dynamic cluster count adaptation

### Enhanced Aggregation
- Adaptive momentum scheduling
- Client contribution weighting
- Outlier detection and filtering

This federated learning system provides a robust, scalable approach to distributed model training with intelligent parameter sharing and aggregation strategies.