import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

# Force unbuffered output so prints appear immediately
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

import flwr as fl
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from app.dataset.chat_logs import load_chat_records
from app.federation.clustering import (
    cluster_aware_average_selected,
    department_level_clustering,
)
from app.federation.flora_aggregation import (
    flora_weighted_aggregate,
    compute_parameter_norm,
)
from app.federation.cluster_monitor import (
    log_cluster_assignment,
    log_aggregation_stats,
    compute_and_log_department_similarities,
)
from app.federation import cluster_config
from app.federation.lora_utils import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DEVICE_MAP,
    DEFAULT_DTYPE,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_LORA_TARGET_MODULES,
    apply_lora_state,
    average_ndarrays,
    clone_numpy_state,
    collect_lora_parameter_names,
    collect_lora_state,
    create_lora_model,
    export_lora_adapter,
    load_adapter_model,
    load_adapter_weights_only,
    lora_state_to_ndarrays,
    ndarrays_to_lora_state,
)


DEFAULT_MAX_SEQ_LENGTH = 256
DEFAULT_BATCH_SIZE = 1
DEFAULT_LOCAL_EPOCHS = 1
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_MAX_RECORDS = 128
DEFAULT_CLIENTS_PER_DEPARTMENT = 10


def _format_training_example(record: Dict[str, object]) -> str:
    user = str(record.get("user", ""))
    assistant = str(record.get("assistant", ""))
    return f"<user>{user}</user>\n<assistant>{assistant}</assistant>"


def _discover_client_paths(
    department: str,
    limit: int,
    dataset_map: Optional[Mapping[str, Path]] = None,
) -> List[Path]:
    custom_entry: Optional[Path] = None
    if dataset_map and department in dataset_map:
        custom_entry = Path(dataset_map[department])

    if custom_entry is not None:
        if custom_entry.is_dir():
            candidates = sorted(custom_entry.glob("*.jsonl"))
        else:
            candidates = [custom_entry]
    else:
        root = Path("app/dataset") / f"{department}_personal_clients"
        candidates = sorted(root.glob("*.jsonl")) if root.exists() else []

    if not candidates:
        raise FileNotFoundError(
            f"No client data files found for department '{department}'. "
            "Provide dataset_map entries or ensure personal client JSONL files exist."
        )

    if limit > 0:
        return candidates[:limit]
    return candidates


class DepartmentLoraClient(fl.client.NumPyClient):
    def __init__(
        self,
        department: str,
        *,
        base_model: str = DEFAULT_BASE_MODEL,
        r: int = DEFAULT_LORA_R,
        alpha: int = DEFAULT_LORA_ALPHA,
        dropout: float = DEFAULT_LORA_DROPOUT,
        target_modules: Optional[Sequence[str]] = DEFAULT_LORA_TARGET_MODULES,
        dtype: str = DEFAULT_DTYPE,
        device_map: str = DEFAULT_DEVICE_MAP,
        batch_size: int = DEFAULT_BATCH_SIZE,
        local_epochs: int = DEFAULT_LOCAL_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        max_records: int = DEFAULT_MAX_RECORDS,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        export_dir: Optional[Path] = None,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[object] = None,
        data_path: Optional[Path] = None,
        load_in_4bit: bool = False,
    ) -> None:
        self.department = department
        self.base_model = base_model
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        self.dtype = dtype
        self.device_map = device_map
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.max_records = max_records
        self.max_seq_length = max_seq_length
        self.export_dir = export_dir or (Path("results") / "client_exports" / department)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = data_path
        self.load_in_4bit = load_in_4bit


        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.model, self.tokenizer = create_lora_model(
                base_model,
                r,
                alpha,
                dropout,
                target_modules=target_modules,
                dtype=dtype,
                device_map=device_map,
                load_in_4bit=load_in_4bit,
            ) 


        # Device handling - be careful with 4-bit quantized models
        try:
            param_device = next(self.model.parameters()).device
            
            # Check if device is None (can happen with 4-bit quantization)
            if param_device is None:
                param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # For 4-bit models, don't try to move them - they're already optimally placed
            if self.load_in_4bit:
                # 4-bit models are already on the correct device
                self.device = param_device if param_device.type != "meta" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif param_device.type in {"meta", "cpu"} and torch.cuda.is_available():
                self.model.to("cuda")
                self.device = torch.device("cuda")
            elif param_device.type != "meta":
                self.device = param_device
            else:
                self.device = torch.device("cpu")
        except (StopIteration, AttributeError):
            # Model has no parameters or device attribute issue
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



        self.tokenizer.model_max_length = max_seq_length
        self.lora_parameter_names = collect_lora_parameter_names(self.model)
        self.training_dir = Path("results") / "client_runs" / department
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.model.train()

    def get_properties(self, config: Dict[str, fl.common.Scalar]) -> Dict[str, fl.common.Scalar]:
        return {"department": self.department}

    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        state = collect_lora_state(self.model, self.lora_parameter_names)
        return lora_state_to_ndarrays(state, self.lora_parameter_names)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        if parameters:
            state = ndarrays_to_lora_state(self.lora_parameter_names, parameters, self.device)
            apply_lora_state(self.model, state)

        local_epochs = int(config.get("local_epochs", self.local_epochs))
        learning_rate = float(config.get("learning_rate", self.learning_rate))

        if "max_records" in config:
            self.max_records = int(config["max_records"])
        if "max_seq_length" in config:
            self.max_seq_length = int(config["max_seq_length"])

        records = self._load_records()
        if not records:
            state = collect_lora_state(self.model, self.lora_parameter_names)
            arrays = lora_state_to_ndarrays(state, self.lora_parameter_names)
            metrics = {"department": self.department, "train_loss": 0.0}
            return arrays, 0, metrics

        dataset = self._build_dataset(records)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        training_args = TrainingArguments(
            output_dir=str(self.training_dir),
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=local_epochs,
            learning_rate=learning_rate,
            logging_steps=5,  # Log frequently so you see it moving
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            disable_tqdm=False,  # Re-enable progress bar so you see it working!
            logging_first_step=True,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        train_output = trainer.train()
        train_metrics = train_output.metrics or {}
        loss = float(train_metrics.get("train_loss", train_output.training_loss))

        # Clear trainer to free GPU memory
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        state = collect_lora_state(self.model, self.lora_parameter_names)
        arrays = lora_state_to_ndarrays(state, self.lora_parameter_names)

        # Export client model for client-side evaluation
        # Use CPU to avoid GPU memory issues
        export_lora_adapter(
            self.base_model,
            self.lora_parameter_names,
            arrays,
            self.export_dir,
            r=self.r,
            alpha=self.alpha,
            dropout=self.dropout,
            target_modules=self.target_modules,
            dtype=self.dtype,
            device_map="cpu",  # Force CPU for export
        )
        
        metrics = {"department": self.department, "train_loss": loss}
        return arrays, len(records), metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> tuple[float, int, Dict[str, fl.common.Scalar]]:
        if parameters:
            state = ndarrays_to_lora_state(self.lora_parameter_names, parameters, self.device)
            apply_lora_state(self.model, state)

        records = self._load_records()
        if not records:
            return 0.0, 0, {"department": self.department}

        dataset = self._build_dataset(records)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        loader = DataLoader(dataset, batch_size=self.batch_size)

        total_loss = 0.0
        total_tokens = 0
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                labels = batch["input_ids"].clone()
                outputs = self.model(**batch, labels=labels)
                loss = outputs.loss
                tokens = labels.numel()
                total_loss += loss.item() * tokens
                total_tokens += tokens

        self.model.train()
        avg_loss = total_loss / total_tokens if total_tokens else 0.0
        return avg_loss, len(records), {"department": self.department, "val_loss": avg_loss}

    def _build_dataset(self, records: List[Dict[str, object]]) -> Dataset:
        texts = [_format_training_example(record) for record in records]
        dataset = Dataset.from_dict({"text": texts})

        def tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
            encodings = self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
            )
            return encodings

        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
        return tokenized

    def _load_records(self) -> List[Dict[str, object]]:
        if self.data_path:
            return self._load_jsonl_records(self.data_path)
        dept_dir = Path("app/dataset") / f"{self.department}_personal_clients"
        records = []
        if dept_dir.exists():
            for jsonl_file in dept_dir.glob("*.jsonl"):
                records.extend(self._load_jsonl_records(jsonl_file))
        return records[:self.max_records]

    def _load_jsonl_records(self, path: Path) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        if not path.exists():
            return records
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(records) >= self.max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                user = str(data.get("text") or "").strip()
                assistant = str(data.get("response") or "").strip()
                if not user or not assistant:
                    continue
                records.append({"user": user, "assistant": assistant})
        return records


def simulate_sequential_training(
    departments: Sequence[str],
    rounds: int,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    r: int = DEFAULT_LORA_R,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
    target_modules: Optional[Sequence[str]] = DEFAULT_LORA_TARGET_MODULES,
    dtype: str = DEFAULT_DTYPE,
    device_map: str = DEFAULT_DEVICE_MAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_records: int = DEFAULT_MAX_RECORDS,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    global_mixing: float = 0.1,
    dataset_map: Optional[Mapping[str, Path]] = None,
    num_clusters: int = 2,
    clients_per_department: int = DEFAULT_CLIENTS_PER_DEPARTMENT,
    enable_dept_clustering: bool = True,
    load_in_4bit: bool = False,
) -> None:
    """
    Enhanced federated training with department-level clustering and FLoRA aggregation.
    
    Parameters
    ----------
    departments : Sequence[str]
        List of department names
    rounds : int
        Number of training rounds
    enable_dept_clustering : bool
        Whether to enable department-level clustering (default: True)
    ... (other parameters same as before)
    """
    print(f"\n{'='*80}")
    print(f"Starting Federated Training with Department-Level Clustering")
    print(f"{'='*80}")
    print(f"Departments: {', '.join(departments)}")
    print(f"Rounds: {rounds}")
    print(f"Clients per department: {clients_per_department}")
    print(f"Department clustering: {'ENABLED' if enable_dept_clustering else 'DISABLED'}")
    print(f"LoRA config: r={r}, alpha={alpha}, dropout={dropout}")
    print(f"{'='*80}\n")
    
    # Initialize model and parameters
    print("Initializing shared model and tokenizer...", flush=True)
    shared_model, shared_tokenizer = create_lora_model(
        base_model,
        r,
        alpha,
        dropout,
        target_modules=target_modules,
        dtype=dtype,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
    )
    print("✓ Model initialized successfully", flush=True)
    
    print("Collecting LoRA parameter names...", flush=True)
    names = collect_lora_parameter_names(shared_model)
    initial_state = lora_state_to_ndarrays(collect_lora_state(shared_model, names), names)
    print(f"✓ Found {len(names)} LoRA parameters\n", flush=True)

    # Load department-specific adapters if they exist
    adapters = {}
    adapter_file = Path("app/lora/lora_adapter.json")
    if adapter_file.exists():
        with adapter_file.open("r", encoding="utf-8") as f:
            adapters = json.load(f)

    department_states: Dict[str, List[np.ndarray]] = {}
    previous_dept_states: Dict[str, List[np.ndarray]] = {}  
    
    for dept in departments:
        adapter_model = adapters.get(dept)
        if adapter_model:
            print(f"  Loading pre-trained adapter for {dept}: {adapter_model}")
            try:
                adapter_state = load_adapter_weights_only(
                    adapter_model,
                    names,
                    device="cpu",
                )
                if adapter_state is not None:
                    department_states[dept] = adapter_state
                    print(f"  ✓ Loaded adapter for {dept}")
                else:
                    print(f"  ⚠ Could not load adapter for {dept}, using initial state")
                    department_states[dept] = clone_numpy_state(initial_state)
            except Exception as e:
                print(f"  ⚠ Error loading adapter for {dept}: {e}")
                department_states[dept] = clone_numpy_state(initial_state)
        else:
            department_states[dept] = clone_numpy_state(initial_state)
        
        previous_dept_states[dept] = clone_numpy_state(department_states[dept])

    # Indices for LoRA A matrices only (where we apply clustering/mixing)
    a_indices = [index for index, param_name in enumerate(names) if "lora_A" in param_name]

    adapter_root = Path("results") / "adapters"
    adapter_root.mkdir(parents=True, exist_ok=True)
    
    metadata_dir = Path(cluster_config.CLUSTER_METADATA_DIR)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Discover client paths for each department
    department_client_paths: Dict[str, List[Path]] = {}
    for dept in departments:
        department_client_paths[dept] = _discover_client_paths(
            dept,
            clients_per_department,
            dataset_map,
        )

    # Main training loop
    for round_index in range(1, rounds + 1):
        print(f"\n{'#'*80}")
        print(f"# Round {round_index}/{rounds}")
        print(f"{'#'*80}\n")
        
        # ===== PHASE 1: CLIENT TRAINING WITHIN DEPARTMENTS =====
        print(f"Phase 1: Training clients within each department...")
        
        for department in departments:
            client_paths = department_client_paths.get(department, [])
            if not client_paths:
                print(f"  {department}: No client data found, skipping")
                continue

            print(f"  {department}: Training {len(client_paths)} clients...")
            
            client_states: List[List[np.ndarray]] = []
            client_weights: List[int] = []

            # Progress bar for client training
            pbar = tqdm(
                enumerate(client_paths),
                total=len(client_paths),
                desc=f"  {department}",
                unit="client",
                leave=True,
            )
            
            for client_idx, client_path in pbar:
                pbar.set_postfix({"file": client_path.name[:20]})
                params = clone_numpy_state(department_states[department])
                client = DepartmentLoraClient(
                    department=department,
                    base_model=base_model,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    target_modules=target_modules,
                    dtype=dtype,
                    device_map=device_map,
                    batch_size=batch_size,
                    local_epochs=local_epochs,
                    learning_rate=learning_rate,
                    max_records=max_records,
                    max_seq_length=max_seq_length,
                    export_dir=Path("results") / "client_exports" / department / f"round_{round_index}_client_{client_idx}",
                    model=shared_model,
                    tokenizer=shared_tokenizer,
                    data_path=client_path,
                    load_in_4bit=load_in_4bit,
                )
                arrays, num_examples, metrics = client.fit(
                    params,
                    {
                        "round": round_index,
                        "local_epochs": local_epochs,
                        "learning_rate": learning_rate,
                    },
                )
                client_states.append(clone_numpy_state(arrays))
                client_weights.append(num_examples if num_examples > 0 else 1)
                
                # Clean up and free GPU memory after each client
                del client
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


            if not client_states:
                continue

            # ===== PHASE 2: INTRA-DEPARTMENT AGGREGATION (Client-level) =====
            # Use cluster-aware aggregation for clients within department
            aggregated_per_item, client_clusters = cluster_aware_average_selected(
                list(zip(client_states, client_weights)),
                a_indices,
                num_clusters=num_clusters,
                max_dim=4096,
                max_iter=25,
                random_state=round_index,
            )

            # Apply global mixing (blend cluster averages with local updates)
            if aggregated_per_item:
                for idx, agg_a in enumerate(aggregated_per_item):
                    if not agg_a:
                        continue
                    state = client_states[idx]
                    if global_mixing > 0.0 and len(client_states) > 1:
                        mix = float(global_mixing)
                        for position, index in enumerate(a_indices):
                            local_arr = state[index]
                            blended = (1.0 - mix) * local_arr + mix * agg_a[position]
                            state[index] = blended.astype(np.float32, copy=False)
                    else:
                        for position, index in enumerate(a_indices):
                            state[index] = np.array(agg_a[position], copy=True)
                    client_states[idx] = state

            # Aggregate all clients using FLoRA method
            prev_state = previous_dept_states.get(department)
            if prev_state is not None and cluster_config.FLORA_USE_RESIDUAL:
                department_states[department] = flora_weighted_aggregate(
                    list(zip(client_states, client_weights)),
                    previous_global=prev_state,
                    momentum=cluster_config.FLORA_MOMENTUM,
                    use_residual=True,
                )
            else:
                department_states[department] = average_ndarrays(
                    list(zip(client_states, client_weights))
                )
            
            # Log statistics
            total_examples = sum(client_weights)
            avg_loss = 0.0  # Would need to track from metrics
            param_norm = compute_parameter_norm(department_states[department])
            
            if cluster_config.ENABLE_CLUSTER_LOGGING:
                log_aggregation_stats(
                    round_index,
                    department,
                    len(client_states),
                    avg_loss,
                    param_norm,
                    metadata_dir if cluster_config.SAVE_CLUSTER_METADATA else None,
                )
        
        # ===== PHASE 3: DEPARTMENT-LEVEL CLUSTERING =====
        if enable_dept_clustering and cluster_config.DEPT_CLUSTERING_ENABLED and len(departments) > 1:
            print(f"\nPhase 2: Clustering departments based on LoRA similarity...")
            
            # Compute and log department similarities
            if cluster_config.ENABLE_CLUSTER_LOGGING:
                compute_and_log_department_similarities(
                    department_states,
                    round_index,
                    metadata_dir if cluster_config.SAVE_CLUSTER_METADATA else None,
                )
            
            # Cluster departments
            dept_clusters, dept_to_cluster = department_level_clustering(
                department_states,
                a_indices,
                num_clusters=None,  # Auto-determine using elbow method
                max_clusters=cluster_config.DEPT_MAX_CLUSTERS,
                max_dim=cluster_config.MAX_EMBEDDING_DIM,
                random_state=cluster_config.RANDOM_SEED + round_index,
            )
            
            # Log cluster assignments
            if cluster_config.ENABLE_CLUSTER_LOGGING:
                log_cluster_assignment(
                    round_index,
                    "department",
                    dept_clusters,
                    metadata_dir if cluster_config.SAVE_CLUSTER_METADATA else None,
                )
            
            # ===== PHASE 4: INTER-DEPARTMENT AGGREGATION (Within clusters) =====
            print(f"\nPhase 3: Aggregating departments within clusters...")
            
            cluster_aggregated_states: Dict[int, List[np.ndarray]] = {}
            
            for cluster_id, dept_list in dept_clusters.items():
                if len(dept_list) == 1:
                    # Single department in cluster, no aggregation needed
                    cluster_aggregated_states[cluster_id] = clone_numpy_state(
                        department_states[dept_list[0]]
                    )
                else:
                    # Multiple departments in cluster, aggregate them
                    print(f"  Cluster {cluster_id}: {', '.join(dept_list)}")
                    
                    cluster_items = []
                    for dept in dept_list:
                        # Weight by parameter norm as proxy for data quantity
                        weight = max(1, int(compute_parameter_norm(department_states[dept])))
                        cluster_items.append((department_states[dept], weight))
                    
                    # Use FLoRA aggregation for cluster
                    cluster_aggregated_states[cluster_id] = flora_weighted_aggregate(
                        cluster_items,
                        previous_global=None,  # No previous for cluster aggregation
                        momentum=0.0,  # No momentum at cluster level
                        use_residual=False,
                    )
            
            # Update department states with cluster-aggregated parameters
            mixing_ratio = cluster_config.DEPT_CLUSTER_MIXING
            for dept in departments:
                cluster_id = dept_to_cluster[dept]
                cluster_avg = cluster_aggregated_states[cluster_id]
                
                # Mix department state with cluster average
                if mixing_ratio > 0.0 and len(dept_clusters[cluster_id]) > 1:
                    mixed_state = []
                    for local, cluster in zip(department_states[dept], cluster_avg):
                        blended = (
                            (1.0 - mixing_ratio) * local.astype(np.float32, copy=False)
                            + mixing_ratio * cluster.astype(np.float32, copy=False)
                        )
                        mixed_state.append(blended)
                    department_states[dept] = mixed_state
        
        # Save previous states for next round's FLoRA residual calculation
        for dept in departments:
            previous_dept_states[dept] = clone_numpy_state(department_states[dept])
        
        # ===== PHASE 5: EXPORT DEPARTMENT ADAPTERS =====
        print(f"\nPhase 4: Exporting department adapters...")
        for department in departments:
            export_lora_adapter(
                base_model,
                names,
                department_states[department],
                adapter_root / department,
                r=r,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules,
                dtype=dtype,
                device_map=device_map,
            )
            print(f"  Exported: {department}")
        
        print(f"\nRound {round_index} complete!\n")
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"Final adapters saved to: {adapter_root}")
    if cluster_config.SAVE_CLUSTER_METADATA:
        print(f"Cluster metadata saved to: {metadata_dir}")
    print(f"{'='*80}\n")



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", nargs="+")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--global-mix", type=float, default=0.1)
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--local-epochs", type=int, default=DEFAULT_LOCAL_EPOCHS)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--device-map", default=DEFAULT_DEVICE_MAP)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--personal-root", type=Path)
    parser.add_argument("--num-clusters", type=int, default=2)
    parser.add_argument("--clients-per-dept", type=int, default=DEFAULT_CLIENTS_PER_DEPARTMENT)
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit quantization")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    
    participants = ["engineering", "finance", "customer_support", "hr"]
    simulate_sequential_training(
        participants,
        rounds=args.rounds,
        global_mixing=args.global_mix,
        learning_rate=args.learning_rate,
        local_epochs=args.local_epochs,
        max_records=args.max_records,
        max_seq_length=args.max_seq_length,
        base_model=args.base_model,
        device_map=args.device_map,
        dtype=args.dtype,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        batch_size=args.batch_size,
        num_clusters=args.num_clusters,
        clients_per_department=args.clients_per_dept,
        enable_dept_clustering=True,
        load_in_4bit=args.load_in_4bit,
    )

# Example:
# source env/bin/activate && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python app/federation/department_client.py --rounds 1 --clients-per-dept 3
# source env/bin/activate && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python app/federation/department_client.py --rounds 1 --clients-per-dept 10


# mac command:
    # python3 -m app.federation.department_client \
#   --personal-root app/dataset/personal_clients \
#   --rounds 3 \
#   --global-mix 0.2 \
#   --max-records 128 \
#   --max-seq-length 256 \
#   --learning-rate 0.0002 \
#   --local-epochs 1 \
#   --batch-size 1 \
#   --device-map auto \
#   --num-clusters 3



