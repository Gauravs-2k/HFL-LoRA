from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def compute_update_residuals(
    current_params: Sequence[np.ndarray],
    previous_params: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """
    Compute residuals (deltas) between current and previous parameters.
    """
    residuals = []
    for curr, prev in zip(current_params, previous_params):
        delta = curr.astype(np.float32, copy=False) - prev.astype(np.float32, copy=False)
        residuals.append(delta)
    return residuals


def flora_weighted_aggregate(
    items: Sequence[Tuple[Sequence[np.ndarray], int]],
    previous_global: Optional[Sequence[np.ndarray]] = None,
    momentum: float = 0.9,
    use_residual: bool = True,
) -> List[np.ndarray]:
    """
    FLoRA-inspired weighted aggregation with residual accumulation.
    """
    if not items:
        return []
    
    total_weight = float(sum(weight for _, weight in items))
    if total_weight <= 0.0:
        return [np.array(arr, copy=True) for arr in items[0][0]]
    
    # Method 1: Residual-based aggregation (FLoRA approach)
    if use_residual and previous_global is not None:
        # Compute weighted average of residuals
        residual_accumulators = [
            np.zeros_like(arr, dtype=np.float32) for arr in items[0][0]
        ]
        
        for params, weight in items:
            residuals = compute_update_residuals(params, previous_global)
            frac = float(weight) / total_weight
            for idx, residual in enumerate(residuals):
                residual_accumulators[idx] += residual * frac
        
        # Apply momentum and add residuals to previous global
        aggregated = []
        for prev, residual_avg in zip(previous_global, residual_accumulators):
            # Momentum: new = momentum * old_update + (1-momentum) * new_update
            # For first round, this simplifies to just adding the residual
            new_param = prev.astype(np.float32, copy=False) + (1.0 - momentum) * residual_avg
            aggregated.append(new_param)
        
        return aggregated
    
    # Method 2: Direct weighted averaging (fallback)
    else:
        accumulators = [
            np.zeros_like(arr, dtype=np.float32) for arr in items[0][0]
        ]
        
        for params, weight in items:
            frac = float(weight) / total_weight
            for idx, param in enumerate(params):
                accumulators[idx] += param.astype(np.float32, copy=False) * frac
        
        return accumulators


def stack_lora_matrices(
    lora_a_list: Sequence[np.ndarray],
    lora_b_list: Sequence[np.ndarray],
    weights: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stack and aggregate LoRA A and B matrices properly.
    """
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights] if total_weight > 0 else weights
    
    # Aggregate A matrices
    a_agg = np.zeros_like(lora_a_list[0], dtype=np.float32)
    for a_matrix, w in zip(lora_a_list, normalized_weights):
        a_agg += a_matrix.astype(np.float32, copy=False) * w
    
    # Aggregate B matrices
    b_agg = np.zeros_like(lora_b_list[0], dtype=np.float32)
    for b_matrix, w in zip(lora_b_list, normalized_weights):
        b_agg += b_matrix.astype(np.float32, copy=False) * w
    
    return a_agg, b_agg


def momentum_based_mixing(
    local_params: Sequence[np.ndarray],
    global_params: Sequence[np.ndarray],
    momentum: float = 0.5,
) -> List[np.ndarray]:
    """
    Mix local and global parameters with momentum.
    """
    mixed = []
    for local, global_p in zip(local_params, global_params):
        blended = (
            momentum * local.astype(np.float32, copy=False)
            + (1.0 - momentum) * global_p.astype(np.float32, copy=False)
        )
        mixed.append(blended)
    return mixed


def compute_parameter_norm(params: Sequence[np.ndarray]) -> float:
    """
    Compute L2 norm of parameters (useful for drift monitoring).
    """
    total_norm_sq = 0.0
    for p in params:
        total_norm_sq += np.sum(p.astype(np.float32, copy=False) ** 2)
    return float(np.sqrt(total_norm_sq))


def compute_cosine_similarity(
    params1: Sequence[np.ndarray],
    params2: Sequence[np.ndarray],
) -> float:
    """
    Compute cosine similarity between two parameter sets.
    """
    dot_product = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0
    
    for p1, p2 in zip(params1, params2):
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
    # Clamp to [0, 1] to handle numerical errors
    return float(max(0.0, min(1.0, similarity)))
