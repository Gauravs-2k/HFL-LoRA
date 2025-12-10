# Model Directory

This directory contains model management scripts, configuration files, and utilities for handling LoRA adapters in the federated learning system.

## Configuration Files

### Model Configurations

#### `dept_lora_base.json`
Contains HuggingFace repository paths for base department-specific LoRA adapters.

```json
{
  "finance": ["Gaurav2k/qwen1.5-1.8b-chat-finance"],
  "hr": ["Gaurav2k/qwen1.5-1.8b-chat-hr"],
  "engineering": ["Gaurav2k/qwen1.5-1.8b-chat-engineering"],
  "it_support": ["Gaurav2k/qwen1.5-1.8b-chat-it_support"]
}
```

#### `dept_lora_federated.json`
Maps departments to their federated training adapter paths (post-training results).

```json
{
  "finance": ["results/adapters/finance"],
  "hr": ["results/adapters/hr"],
  "engineering": ["results/adapters/engineering"],
  "it_support": ["results/adapters/customer_support"]
}
```

#### `client_lora_federated.json`
Comprehensive mapping of client adapters across all federated learning rounds.

**Structure:**
```json
{
  "finance": {
    "round_1": ["results/client_exports/finance/round_1_client_0", ...],
    "round_2": ["results/client_exports/finance/round_2_client_0", ...],
    ...
  },
  "hr": {...},
  "engineering": {...},
  "it_support": {...}
}
```

**Coverage:** 10 rounds × 3 clients × 4 departments = 120 client adapter paths

## Scripts

### Model Management

#### `download.py`
Downloads and sets up LoRA models with automatic target module detection.

**Usage:**
```bash
python app/model/download.py \
  --model-name "Qwen/Qwen1.5-1.8B-Chat" \
  --output-dir "qwen_lora" \
  --dtype "auto" \
  --device-map "auto"

python app/model/download.py \
  --model-name "Qwen/Qwen1.5-1.8B-Chat" \
  --target-modules "q_proj" "v_proj" "k_proj" "o_proj"
```

#### `upload_adapters.py`
Uploads trained LoRA adapters to HuggingFace Hub.

**Usage:**
```bash
export HF_TOKEN="your_huggingface_token"
python app/model/upload_adapters.py
```

**What it does:**
- Creates public repositories for each department
- Uploads LoRA adapters to `Gaurav2k/qwen1.5-1.8b-chat-{department}`

### Inference & Serving

#### `inference.py`
FastAPI-based inference server for LoRA models with adapter switching.

**Usage:**
```bash
python app/model/inference.py \
  --base-model "Qwen/Qwen1.5-1.8B-Chat" \
  --peft-dir "results/adapters" \
  --port 8000 \
  --dtype "auto"


python app/model/inference.py \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --device-map "cuda:0"
```

**API Endpoints:**
- `POST /generate`: Generate text with optional department adapter
- `GET /adapters`: List loaded adapters
- `POST /load_adapter`: Load specific department adapter
- `POST /unload_adapter`: Unload adapter to free memory

### Utilities

#### `gpu_clean.py`
GPU memory cleanup and model unloading utilities.

**Usage:**
```bash
python app/model/gpu_clean.py
```
