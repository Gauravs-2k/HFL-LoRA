"""
FLoRA-inspired aggregation methods for federated learning.

This module implements robust aggregation strategies inspired by FLoRA
(Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations)
to reduce client drift and improve convergence in federated learning scenarios.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def compute_update_residuals(
    current_params: Sequence[np.ndarray],
    previous_params: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """
    Compute residuals (deltas) between current and previous parameters.
    
    This is key to FLoRA's approach: instead of directly aggregating parameters,
    we aggregate the updates/residuals to reduce drift.
    
    Parameters
    ----------
    current_params : Sequence[np.ndarray]
        Current parameter arrays
    previous_params : Sequence[np.ndarray]
        Previous parameter arrays (from last round)
    
    Returns
    -------
    List[np.ndarray]
        Residual arrays for each parameter
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
    
    This method implements the core principles of FLoRA:
    1. Residual-based aggregation to reduce drift
    2. Proper mathematical weighting to avoid aggregation noise
    3. Momentum-based updates for stability
    
    Parameters
    ----------
    items : Sequence[Tuple[Sequence[np.ndarray], int]]
        List of (parameters, weight) tuples from each client
    previous_global : Optional[Sequence[np.ndarray]]
        Previous global parameters for residual calculation
    momentum : float
        Momentum coefficient (0.0 = no momentum, 0.9 = high momentum)
    use_residual : bool
        Whether to use residual-based aggregation
    
    Returns
    -------
    List[np.ndarray]
        Aggregated parameters
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
    
    In LoRA, the adaptation is W = BA where B is (out_dim, r) and A is (r, in_dim).
    Proper aggregation should maintain the low-rank structure.
    
    Parameters
    ----------
    lora_a_list : Sequence[np.ndarray]
        List of LoRA A matrices
    lora_b_list : Sequence[np.ndarray]
        List of LoRA B matrices
    weights : Sequence[float]
        Normalized weights for each client
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Aggregated (A, B) matrices
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
    
    This creates a soft update that prevents sudden parameter changes
    and reduces oscillation during training.
    
    Parameters
    ----------
    local_params : Sequence[np.ndarray]
        Local client parameters
    global_params : Sequence[np.ndarray]
        Global aggregated parameters
    momentum : float
        Mixing ratio (0.0 = full global, 1.0 = full local)
    
    Returns
    -------
    List[np.ndarray]
        Mixed parameters
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
    
    Parameters
    ----------
    params : Sequence[np.ndarray]
        Parameter arrays
    
    Returns
    -------
    float
        L2 norm
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
    
    This is useful for measuring similarity between departments
    or tracking how much parameters have changed.
    
    Parameters
    ----------
    params1 : Sequence[np.ndarray]
        First parameter set
    params2 : Sequence[np.ndarray]
        Second parameter set
    
    Returns
    -------
    float
        Cosine similarity in [0, 1]
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
