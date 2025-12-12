from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.model.inference import parse_dtype

DEFAULT_BASE_MODEL = "Qwen/Qwen1.5-1.8B-Chat"
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES: Optional[Sequence[str]] = ["q_proj", "v_proj"]
DEFAULT_DTYPE = "auto"
DEFAULT_DEVICE_MAP = "auto"  


def create_tokenizer(base_model: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    return tokenizer


def create_lora_model(
    base_model: str = DEFAULT_BASE_MODEL,
    r: int = DEFAULT_LORA_R,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
    *,
    target_modules: Optional[Sequence[str]] = DEFAULT_LORA_TARGET_MODULES,
    dtype: str = DEFAULT_DTYPE,
    device_map: str = DEFAULT_DEVICE_MAP,
    load_in_4bit: bool = False,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    tokenizer = create_tokenizer(base_model)
    kwargs: Dict[str, object] = {"trust_remote_code": True}
    torch_dtype = parse_dtype(dtype)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if device_map.lower() != "none":
        kwargs["device_map"] = device_map
    if load_in_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = None  # Let transformers handle default 4bit config
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    model.resize_token_embeddings(len(tokenizer))
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules) if target_modules is not None else None,
    )
    peft_model = get_peft_model(model, config)
    return peft_model, tokenizer


def _resolve_state_tensor(full_state: Dict[str, torch.Tensor], name: str) -> torch.Tensor:
    if name in full_state:
        return full_state[name]
    suffix = name.split("base_model.", 1)[-1]
    for key, tensor in full_state.items():
        if key.endswith(suffix):
            return tensor
    raise KeyError(name)


def load_adapter_model(
    adapter_model: str,
    base_model: str = DEFAULT_BASE_MODEL,
    r: int = DEFAULT_LORA_R,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
    *,
    target_modules: Optional[Sequence[str]] = DEFAULT_LORA_TARGET_MODULES,
    dtype: str = DEFAULT_DTYPE,
    device_map: str = DEFAULT_DEVICE_MAP,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    model, tokenizer = create_lora_model(
        base_model,
        r,
        alpha,
        dropout,
        target_modules=target_modules,
        dtype=dtype,
        device_map=device_map,
    )
    peft_model = PeftModel.from_pretrained(model, adapter_model)
    return peft_model, tokenizer


def load_adapter_weights_only(
    adapter_model: str,
    parameter_names: Sequence[str],
    device: str = "cpu",
) -> Optional[List[np.ndarray]]:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as load_safetensors
    import os
    
    try:
        adapter_path = hf_hub_download(
            repo_id=adapter_model,
            filename="adapter_model.safetensors",
        )
        adapter_state = load_safetensors(adapter_path, device=device)
    except Exception:
        try:
            adapter_path = hf_hub_download(
                repo_id=adapter_model,
                filename="adapter_model.bin",
            )
            adapter_state = torch.load(adapter_path, map_location=device)
        except Exception as e:
            print(f"    Could not download adapter: {e}")
            return None
    
    arrays: List[np.ndarray] = []
    for name in parameter_names:
        matched = False
        for key, tensor in adapter_state.items():
            name_suffix = name.split("base_model.model.")[-1]
            key_suffix = key.replace("base_model.model.", "")
            if name_suffix == key_suffix or name.endswith(key) or key.endswith(name_suffix):
                arrays.append(tensor.cpu().numpy().astype(np.float32))
                matched = True
                break
        if not matched:
            return None
    
    return arrays if len(arrays) == len(parameter_names) else None


def collect_lora_parameter_names(model: torch.nn.Module) -> List[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]


def collect_lora_state(model: torch.nn.Module, names: Sequence[str]) -> OrderedDict[str, torch.Tensor]:
    state = OrderedDict()
    full_state = model.state_dict()
    for name in names:
        tensor = _resolve_state_tensor(full_state, name)
        state[name] = tensor.detach().clone()
    return state


def apply_lora_state(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    target_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in state.items():
            target = target_params[name]
            target.copy_(tensor.to(target.device))


def lora_state_to_ndarrays(state: Dict[str, torch.Tensor], names: Sequence[str]) -> List[np.ndarray]:
    arrays: List[np.ndarray] = []
    for name in names:
        tensor = state[name].detach().cpu().numpy().astype(np.float32, copy=True)
        arrays.append(tensor)
    return arrays


def ndarrays_to_lora_state(names: Sequence[str], arrays: Sequence[np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, array in zip(names, arrays):
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device)
        state[name] = tensor
    return state


def clone_numpy_state(arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [np.array(array, copy=True) for array in arrays]


def average_ndarrays(items: Sequence[Tuple[Sequence[np.ndarray], int]]) -> List[np.ndarray]:
    total_weight = sum(weight for _, weight in items)
    if total_weight == 0:
        return [np.array(arr, copy=True) for arr in items[0][0]]
    accumulators = [np.zeros_like(arr, dtype=np.float32) for arr in items[0][0]]
    for arrays, weight in items:
        fraction = weight / total_weight
        for index, array in enumerate(arrays):
            accumulators[index] += array.astype(np.float32, copy=False) * fraction
    return accumulators


def export_lora_adapter(
    base_model: str,
    names: Sequence[str],
    arrays: Sequence[np.ndarray],
    output_dir: Path,
    *,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Optional[Sequence[str]] = DEFAULT_LORA_TARGET_MODULES,
    dtype: str = DEFAULT_DTYPE,
    device_map: str = DEFAULT_DEVICE_MAP,
) -> None:
    """
    Export ONLY LoRA adapter weights (not the full model).
    This saves ~90% disk space compared to saving the full model.
    """
    import json
    from safetensors.torch import save_file
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lora_state_dict = {}
    for name, array in zip(names, arrays):
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
        clean_name = name.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")
        lora_state_dict[clean_name] = tensor
    
    save_file(lora_state_dict, output_dir / "adapter_model.safetensors")
    
    # Save adapter configuration
    adapter_config = {
        "base_model_name_or_path": base_model,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": list(target_modules) if target_modules else None,
        "bias": "none",
        "fan_in_fan_out": False,
        "modules_to_save": None,
    }
    
    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)
    
    # Save tokenizer (this is small, ~1MB)
    tokenizer = create_tokenizer(base_model)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✓ Saved LoRA adapter ({len(lora_state_dict)} params, ~{sum(t.numel() for t in lora_state_dict.values()) * 4 / 1024 / 1024:.1f}MB)")



def compute_lora_similarity(
    state1: Sequence[np.ndarray],
    state2: Sequence[np.ndarray],
) -> float:
    """
    Compute cosine similarity between two LoRA parameter states.
    
    Parameters
    ----------
    state1 : Sequence[np.ndarray]
        First LoRA state
    state2 : Sequence[np.ndarray]
        Second LoRA state
    
    Returns
    -------
    float
        Cosine similarity in [0, 1]
    """
    dot_product = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0
    
    for p1, p2 in zip(state1, state2):
        p1_flat = p1.astype(np.float32, copy=False).ravel()
        p2_flat = p2.astype(np.float32, copy=False).ravel()
        
        dot_product += np.dot(p1_flat, p2_flat)
        norm1_sq += np.sum(p1_flat ** 2)
        norm2_sq += np.sum(p2_flat ** 2)
    
    norm1 = np.sqrt(norm1_sq)
    norm2 = np.sqrt(norm2_sq)
    
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(max(0.0, min(1.0, similarity)))


def extract_lora_embedding(
    arrays: Sequence[np.ndarray],
    indices: Optional[Sequence[int]] = None,
    max_dim: int = 4096,
) -> np.ndarray:
    """
    Extract a fixed-size embedding from LoRA parameters for clustering.
    
    Parameters
    ----------
    arrays : Sequence[np.ndarray]
        LoRA parameter arrays
    indices : Optional[Sequence[int]]
        Specific indices to use (e.g., only LoRA A matrices).
        If None, uses all parameters.
    max_dim : int
        Maximum embedding dimension
    
    Returns
    -------
    np.ndarray
        Fixed-size embedding vector
    """
    import math
    
    selected = []
    param_indices = indices if indices is not None else range(len(arrays))
    
    for idx in param_indices:
        if idx < len(arrays):
            w = arrays[idx].astype(np.float32, copy=False).ravel()
            selected.append(w)
    
    if not selected:
        return np.zeros((max_dim,), dtype=np.float32)
    
    vec = np.concatenate(selected, axis=0)
    
    # Downsample if too large
    if vec.size > max_dim:
        step = math.ceil(vec.size / max_dim)
        vec = vec[::step][:max_dim]
    
    return vec

