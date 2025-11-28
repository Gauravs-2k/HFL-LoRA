"""
Monitoring and visualization utilities for clustering in federated learning.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def log_cluster_assignment(
    round_num: int,
    cluster_type: str,
    clusters: Dict[int, List[str]],
    output_dir: Optional[Path] = None,
) -> None:
    """
    Log cluster assignments to console and optionally to file.
    
    Parameters
    ----------
    round_num : int
        Current training round
    cluster_type : str
        Type of clustering ("department" or "client")
    clusters : Dict[int, List[str]]
        Mapping from cluster ID to list of member names
    output_dir : Optional[Path]
        Directory to save cluster metadata
    """
    print(f"\n{'='*60}")
    print(f"Round {round_num} - {cluster_type.capitalize()} Clustering")
    print(f"{'='*60}")
    
    for cluster_id, members in sorted(clusters.items()):
        print(f"Cluster {cluster_id}: {', '.join(members)}")
    
    print(f"{'='*60}\n")
    
    # Save to file if output directory provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = output_dir / f"round_{round_num}_{cluster_type}_clusters.json"
        
        metadata = {
            "round": round_num,
            "type": cluster_type,
            "num_clusters": len(clusters),
            "clusters": {str(k): v for k, v in clusters.items()},
        }
        
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def log_similarity_matrix(
    round_num: int,
    names: Sequence[str],
    similarity_matrix: np.ndarray,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Log similarity matrix between departments/clients.
    
    Parameters
    ----------
    round_num : int
        Current training round
    names : Sequence[str]
        Names of entities (departments or clients)
    similarity_matrix : np.ndarray
        Pairwise similarity matrix (n x n)
    output_dir : Optional[Path]
        Directory to save metadata
    """
    print(f"\n{'='*60}")
    print(f"Round {round_num} - Similarity Matrix")
    print(f"{'='*60}")
    
    # Print header
    name_list = list(names)
    header = "          " + "  ".join(f"{name[:8]:>8}" for name in name_list)
    print(header)
    print("-" * len(header))
    
    # Print matrix
    for i, name in enumerate(name_list):
        row = f"{name[:8]:>8}  "
        row += "  ".join(f"{similarity_matrix[i, j]:>8.3f}" for j in range(len(name_list)))
        print(row)
    
    print(f"{'='*60}\n")
    
    # Save to file
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = output_dir / f"round_{round_num}_similarity.json"
        
        metadata = {
            "round": round_num,
            "names": name_list,
            "similarity_matrix": similarity_matrix.tolist(),
        }
        
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def log_aggregation_stats(
    round_num: int,
    department: str,
    num_clients: int,
    avg_loss: float,
    parameter_norm: float,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Log aggregation statistics for a department.
    
    Parameters
    ----------
    round_num : int
        Current training round
    department : str
        Department name
    num_clients : int
        Number of clients aggregated
    avg_loss : float
        Average training loss
    parameter_norm : float
        L2 norm of aggregated parameters
    output_dir : Optional[Path]
        Directory to save logs
    """
    print(f"Round {round_num} - {department}: "
          f"Clients={num_clients}, Loss={avg_loss:.4f}, Norm={parameter_norm:.2f}")
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / f"round_{round_num}_stats.jsonl"
        
        stats = {
            "round": round_num,
            "department": department,
            "num_clients": num_clients,
            "avg_loss": avg_loss,
            "parameter_norm": parameter_norm,
        }
        
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")


def save_elbow_curve_data(
    round_num: int,
    k_values: List[int],
    inertias: List[float],
    optimal_k: int,
    output_dir: Path,
) -> None:
    """
    Save elbow curve data for visualization.
    
    Parameters
    ----------
    round_num : int
        Current training round
    k_values : List[int]
        List of k values tested
    inertias : List[float]
        Inertia values for each k
    optimal_k : int
        Selected optimal number of clusters
    output_dir : Path
        Directory to save data
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    elbow_file = output_dir / f"round_{round_num}_elbow_curve.json"
    
    data = {
        "round": round_num,
        "k_values": k_values,
        "inertias": inertias,
        "optimal_k": optimal_k,
    }
    
    with elbow_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"  Elbow method: tested k={min(k_values)}-{max(k_values)}, selected k={optimal_k}")


def compute_and_log_department_similarities(
    department_states: Dict[str, List[np.ndarray]],
    round_num: int,
    output_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Compute and log pairwise similarities between departments.
    
    Parameters
    ----------
    department_states : Dict[str, List[np.ndarray]]
        Department LoRA states
    round_num : int
        Current round
    output_dir : Optional[Path]
        Output directory
    
    Returns
    -------
    np.ndarray
        Similarity matrix
    """
    from app.federation.lora_utils import compute_lora_similarity
    
    dept_names = sorted(department_states.keys())
    n = len(dept_names)
    similarity_matrix = np.zeros((n, n), dtype=np.float32)
    
    for i, dept1 in enumerate(dept_names):
        for j, dept2 in enumerate(dept_names):
            if i == j:
                similarity_matrix[i, j] = 1.0
            elif i < j:
                sim = compute_lora_similarity(
                    department_states[dept1],
                    department_states[dept2]
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
    
    log_similarity_matrix(round_num, dept_names, similarity_matrix, output_dir)
    return similarity_matrix
