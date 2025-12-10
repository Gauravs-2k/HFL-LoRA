# Dataset Directory

This directory contains all the data preparation scripts, training datasets, and evaluation datasets for the federated learning project with LoRA adapters.

## Data Files Description

### Training Datasets

#### Department-Level Datasets (`edge_lora_*.jsonl`)
- **Purpose**: Combined training data for each department
- **Format**: JSONL with `{"text": "...", "response": "..."}` entries
- **Usage**: Used for evaluation and department-level model training
- **Files**:
  - `edge_lora_engineering.jsonl` (~2.5MB, ~2,400 samples)
  - `edge_lora_finance.jsonl` (~1.2MB, ~1,200 samples)
  - `edge_lora_hr.jsonl` (~900KB, ~900 samples)
  - `edge_lora_support.jsonl` (~1.8MB, ~1,800 samples)
  - `edge_lora_cusotmer_support.jsonl` (~670KB, ~700 samples)

#### Department Training Data (`dept/*.jsonl`)
- **Purpose**: Large-scale training datasets for department models
- **Format**: JSONL with training examples
- **Size**: ~1.5-1.7MB each, ~3,000 samples per department
- **Usage**: Used for initial department model training

#### Personal Client Datasets (`*_personal_clients/`)
- **Purpose**: Individual client training data for federated learning
- **Structure**: Each department has 10 client files
- **Format**: JSONL with `{"text": "...", "response": "..."}` entries
- **Usage**: Used by `DepartmentLoraClient` during federated training
- **Example**: `engineering_personal_clients/1e230e55efea4edab95db9cb87f6a9cb.jsonl`

### Evaluation Datasets (`test/*.jsonl`)

- **Purpose**: Test datasets for model evaluation
- **Format**: JSONL with `{"prompt": "...", "response": "..."}` entries
- **Size**: 200 samples per department
- **Usage**: Used by evaluation scripts to measure model performance
- **Files**:
  - `ENGINEERING_test.jsonl` (200 samples, algorithm/programming problems)
  - `FINANCE_test.jsonl` (200 samples, financial queries)
  - `HR_test.jsonl` (200 samples, HR/training questions)
  - `IT_SUPPORT_test.jsonl` (200 samples, technical support scenarios)

### Configuration Files

- **`hf_data.json`**: HuggingFace dataset metadata and cache information
- **`personalised_data.json`**: Client-specific training data with user IDs

## Scripts

### Data Generation Scripts

#### `create_edge_lora_data.py`
Creates department-specific training datasets by combining personal data with HuggingFace datasets.

**Usage Examples:**
```bash
python app/dataset/create_edge_lora_data.py \
  --department "finance" \
  --hf-dataset "sweatSmile/FinanceQA" \
  --hf-limit 1000 \
  --per-client-hf-limit 100 \
  --personal-output-dir app/dataset/finance_personal_clients

python app/dataset/create_edge_lora_data.py \
  --department "engineering" \
  --hf-dataset "nvidia/OpenCodeInstruct" \
  --hf-limit 2000 \
  --per-client-hf-limit 100 \
  --personal-output-dir app/dataset/engineering_personal_clients
```

**Parameters:**
- `--department`: Department name (engineering, finance, hr, it_support)
- `--hf-dataset`: HuggingFace dataset to use
- `--hf-limit`: Total samples from HF dataset
- `--per-client-hf-limit`: Samples per client from HF dataset
- `--personal-output-dir`: Directory for client-specific files

#### `generate_datasets.py`
Generates synthetic training data using LLM for department-specific contexts.

**Usage:**
```bash
python app/dataset/generate_datasets.py \
  --department "engineering" \
  --count 1000 \
  --model "gpt-4" \
  --output-dir app/dataset/dept/
```

#### `generate_contexts.py`
Generates context files for each department with domain-specific information.

**Usage:**
```bash
python app/dataset/generate_contexts.py
```

### Utility Scripts

#### `chat_logs.py`
Utilities for logging and managing chat interactions.

**Functions:**
- `append_chat_record()`: Add chat entries to department logs
- `load_chat_records()`: Load chat history for a department
- `list_recorded_departments()`: List departments with chat logs

## Data Flow

1. **Data Creation**: Use `create_edge_lora_data.py` to generate department datasets
2. **Federated Training**: Clients load data from `*_personal_clients/` directories
3. **Model Training**: Department models trained on combined datasets
4. **Evaluation**: Models tested on `test/*.jsonl` datasets
5. **Logging**: Chat interactions logged using `chat_logs.py`

## Usage in Federated Learning

### Client Training
Each `DepartmentLoraClient` loads training data from its department's personal client directory:

```python
dept_dir = Path("app/dataset") / f"{department}_personal_clients"
for jsonl_file in dept_dir.glob("*.jsonl"):
    records.extend(self._load_jsonl_records(jsonl_file))
```

### Evaluation
Evaluation scripts use the `edge_lora_*.jsonl` files:

```python
dataset_path = args.dataset_root / f"edge_lora_{department}.jsonl"
```

## Data Statistics

- **Total Training Samples**: ~12,000+ across all departments
- **Test Samples**: 800 (200 per department)
- **Clients per Department**: 10
- **Federated Rounds**: 10 (typical usage)
- **Departments**: Engineering, Finance, HR, IT Support

## File Formats

### JSONL Format
All data files use JSONL format with one JSON object per line:

```json
{"text": "User query or instruction", "response": "Expected model response"}
```

### Context Files
Department context files contain domain-specific information for data generation.

## Dependencies

- `datasets` (HuggingFace)
- `openai` (for synthetic data generation)
- `pathlib` (file operations)
- `json` (data parsing)
