from __future__ import annotations


import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import silhouette_score


def cluster_aware_average_selected(
    items: Sequence[Tuple[Sequence[np.ndarray], int]],
    indices: Sequence[int],
    *,
    num_clusters: int = 1,
    max_dim: int = 4096,
    max_iter: int = 25,
    tol: float = 1e-4,  # reserved for future stopping criteria
    random_state: Optional[int] = None,
) -> Tuple[List[List[np.ndarray]], List[int]]:
    """
    Cluster-aware aggregation over selected parameters (e.g., LoRA A matrices).

    Parameters
    ----------
    items:
        Sequence of (arrays, weight), where `arrays` is the LoRA state
        and `weight` is typically the number of local examples.
    indices:
        Indices into `arrays` specifying which parameters to aggregate.
    num_clusters:
        Number of clusters for k-means. If <=1, behaves like global FedAvg.
    max_dim:
        Max embedding dimension for clustering (downsamples large vectors).
    max_iter:
        Max k-means iterations.
    tol:
        Currently unused (reserved for future stopping criteria).
    random_state:
        Seed for reproducibility.

    Returns
    -------
    aggregated_per_item:
        List of length len(items); each element is a list of ndarrays
        giving the aggregated parameters for that item's cluster.
    labels:
        List of cluster labels per item (same order as `items`).
    """
    n = len(items)
    if n == 0 or not indices:
        return [], []

    # ---------- No-cluster or trivial cluster cases ----------
    if num_clusters <= 1:
        # Single global FedAvg across all items
        total_weight = float(sum(weight for _, weight in items))
        if total_weight <= 0.0:
            global_avg = [np.array(items[0][0][idx], copy=True) for idx in indices]
        else:
            global_avg = [
                np.zeros_like(items[0][0][idx], dtype=np.float32) for idx in indices
            ]
            for arrays, weight in items:
                frac = float(weight) / total_weight
                for pos, idx in enumerate(indices):
                    global_avg[pos] += arrays[idx].astype(np.float32, copy=False) * frac

        aggregated_per_item = [global_avg for _ in range(n)]
        labels = [0] * n
        return aggregated_per_item, labels

    if num_clusters >= n:
        # Each item gets its own "cluster" (no cross-client mixing)
        aggregated_per_item: List[List[np.ndarray]] = []
        labels: List[int] = []
        for i, (arrays, _) in enumerate(items):
            agg = [np.array(arrays[idx], copy=True) for idx in indices]
            aggregated_per_item.append(agg)
            labels.append(i)
        return aggregated_per_item, labels

    # ---------- Build embeddings from selected LoRA params ----------
    def _build_embedding(arrays: Sequence[np.ndarray]) -> np.ndarray:
        selected: List[np.ndarray] = []
        for idx in indices:
            w = arrays[idx].astype(np.float32, copy=False).ravel()
            selected.append(w)
        if not selected:
            return np.zeros((max_dim,), dtype=np.float32)
        vec = np.concatenate(selected, axis=0)
        if vec.size > max_dim:
            step = math.ceil(vec.size / max_dim)
            vec = vec[::step][:max_dim]
        return vec

    embeddings = [_build_embedding(arrays) for arrays, _ in items]
    X = np.stack(embeddings, axis=0)  # (n, d)

    rng = np.random.RandomState(0 if random_state is None else random_state)

    # ---------- k-means++ initialization ----------
    centroids = np.empty((num_clusters, X.shape[1]), dtype=np.float32)
    first_idx = rng.randint(0, n)
    centroids[0] = X[first_idx]
    for k in range(1, num_clusters):
        dists_sq = np.min(
            ((X[:, None, :] - centroids[None, :k, :]) ** 2).sum(axis=2),
            axis=1,
        )
        probs = dists_sq / (dists_sq.sum() + 1e-12)
        next_idx = rng.choice(n, p=probs)
        centroids[k] = X[next_idx]

    labels = np.zeros(n, dtype=np.int32)

    # ---------- Lloyd's k-means iterations ----------
    for _ in range(max_iter):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dists, axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for k in range(num_clusters):
            mask = labels == k
            if not np.any(mask):
                centroids[k] = X[rng.randint(0, n)]
            else:
                centroids[k] = X[mask].mean(axis=0)

    # ---------- Cluster-wise weighted averaging on selected indices ----------
    clusters: Dict[int, List[Tuple[Sequence[np.ndarray], int]]] = defaultdict(list)
    for i, (arrays, weight) in enumerate(items):
        cid = int(labels[i])
        clusters[cid].append((arrays, weight))

    cluster_avgs: Dict[int, List[np.ndarray]] = {}
    for cid, cluster_items in clusters.items():
        total_weight = float(sum(w for _, w in cluster_items))
        if total_weight <= 0.0:
            cluster_avgs[cid] = [
                np.array(cluster_items[0][0][idx], copy=True) for idx in indices
            ]
            continue

        acc = [
            np.zeros_like(cluster_items[0][0][idx], dtype=np.float32)
            for idx in indices
        ]
        for arrays, weight in cluster_items:
            frac = float(weight) / total_weight
            for pos, idx in enumerate(indices):
                acc[pos] += arrays[idx].astype(np.float32, copy=False) * frac
        cluster_avgs[cid] = acc

    # ---------- Expand back to per-item aggregated params ----------
    aggregated_per_item: List[List[np.ndarray]] = []
    for i in range(n):
        cid = int(labels[i])
        avg_for_cluster = cluster_avgs[cid]
        aggregated_per_item.append([np.array(a, copy=True) for a in avg_for_cluster])

    return aggregated_per_item, labels.tolist()


def determine_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int = 10,
    min_clusters: int = 2,
    method: str = "elbow",
    random_state: Optional[int] = None,
) -> int:
    """
    Determine optimal number of clusters using elbow method or silhouette analysis.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Data embeddings to cluster (n_samples, n_features)
    max_clusters : int
        Maximum number of clusters to try
    min_clusters : int
        Minimum number of clusters to try
    method : str
        Method to use: "elbow" or "silhouette"
    random_state : Optional[int]
        Random seed for reproducibility
    
    Returns
    -------
    int
        Optimal number of clusters
    """
    n_samples = embeddings.shape[0]
    
    # Can't have more clusters than samples
    max_clusters = min(max_clusters, n_samples - 1)
    min_clusters = max(2, min(min_clusters, max_clusters))
    
    if min_clusters >= max_clusters:
        return min_clusters
    
    rng = np.random.RandomState(random_state if random_state is not None else 0)
    
    if method == "silhouette":
        return _silhouette_method(embeddings, min_clusters, max_clusters, rng)
    else:
        return _elbow_method(embeddings, min_clusters, max_clusters, rng)


def _elbow_method(
    embeddings: np.ndarray,
    min_k: int,
    max_k: int,
    rng: np.random.RandomState,
) -> int:
    """
    Use elbow method to find optimal number of clusters.
    
    The elbow method looks for the "elbow" in the curve of inertia
    (sum of squared distances to cluster centers) vs number of clusters.
    """
    inertias = []
    k_range = range(min_k, max_k + 1)
    
    for k in k_range:
        _, inertia = _run_kmeans(embeddings, k, rng, max_iter=50)
        inertias.append(inertia)
    
    # Find elbow using second derivative
    if len(inertias) < 3:
        return min_k
    
    # Normalize inertias to [0, 1]
    inertias_arr = np.array(inertias)
    inertias_norm = (inertias_arr - inertias_arr.min()) / (inertias_arr.max() - inertias_arr.min() + 1e-10)
    
    # Compute second derivative (discrete)
    second_deriv = []
    for i in range(1, len(inertias_norm) - 1):
        d2 = inertias_norm[i - 1] - 2 * inertias_norm[i] + inertias_norm[i + 1]
        second_deriv.append(d2)
    
    # Find the point with maximum curvature (elbow)
    if second_deriv:
        elbow_idx = np.argmax(second_deriv) + 1  # +1 because we started from index 1
        optimal_k = min_k + elbow_idx
    else:
        optimal_k = min_k
    
    return optimal_k


def _silhouette_method(
    embeddings: np.ndarray,
    min_k: int,
    max_k: int,
    rng: np.random.RandomState,
) -> int:
    """
    Use silhouette score to find optimal number of clusters.
    
    Higher silhouette score indicates better-defined clusters.
    """
    best_score = -1.0
    best_k = min_k
    
    for k in range(min_k, max_k + 1):
        labels, _ = _run_kmeans(embeddings, k, rng, max_iter=50)
        
        # Need at least 2 clusters for silhouette score
        if len(np.unique(labels)) < 2:
            continue
        
        try:
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except:
            continue
    
    return best_k


def _run_kmeans(
    X: np.ndarray,
    k: int,
    rng: np.random.RandomState,
    max_iter: int = 50,
) -> Tuple[np.ndarray, float]:
    """
    Run k-means clustering and return labels and inertia.
    
    Returns
    -------
    labels : np.ndarray
        Cluster labels
    inertia : float
        Sum of squared distances to nearest cluster center
    """
    n = X.shape[0]
    
    # k-means++ initialization
    centroids = np.empty((k, X.shape[1]), dtype=np.float32)
    first_idx = rng.randint(0, n)
    centroids[0] = X[first_idx]
    
    for i in range(1, k):
        dists_sq = np.min(
            ((X[:, None, :] - centroids[None, :i, :]) ** 2).sum(axis=2),
            axis=1,
        )
        probs = dists_sq / (dists_sq.sum() + 1e-12)
        next_idx = rng.choice(n, p=probs)
        centroids[i] = X[next_idx]
    
    labels = np.zeros(n, dtype=np.int32)
    
    # Lloyd's algorithm
    for _ in range(max_iter):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dists, axis=1)
        
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centroids[i] = X[mask].mean(axis=0)
            else:
                centroids[i] = X[rng.randint(0, n)]
    
    # Compute inertia
    inertia = 0.0
    for i in range(n):
        inertia += np.sum((X[i] - centroids[labels[i]]) ** 2)
    
    return labels, float(inertia)


def department_level_clustering(
    department_states: Dict[str, List[np.ndarray]],
    indices: Sequence[int],
    *,
    num_clusters: Optional[int] = None,
    max_clusters: int = 5,
    max_dim: int = 4096,
    random_state: Optional[int] = None,
) -> Tuple[Dict[int, List[str]], Dict[str, int]]:
    """
    Cluster departments based on their aggregated LoRA states.
    
    Parameters
    ----------
    department_states : Dict[str, List[np.ndarray]]
        Dictionary mapping department names to their LoRA parameter arrays
    indices : Sequence[int]
        Indices of parameters to use for clustering (e.g., LoRA A matrices)
    num_clusters : Optional[int]
        Number of clusters. If None, will be determined automatically.
    max_clusters : int
        Maximum number of clusters for auto-detection
    max_dim : int
        Maximum embedding dimension
    random_state : Optional[int]
        Random seed
    
    Returns
    -------
    clusters : Dict[int, List[str]]
        Mapping from cluster ID to list of department names
    dept_to_cluster : Dict[str, int]
        Mapping from department name to cluster ID
    """
    if not department_states:
        return {}, {}
    
    dept_names = sorted(department_states.keys())
    n_depts = len(dept_names)
    
    if n_depts == 1:
        return {0: dept_names}, {dept_names[0]: 0}
    
    # Build embeddings for each department
    def _build_dept_embedding(arrays: Sequence[np.ndarray]) -> np.ndarray:
        selected = []
        for idx in indices:
            w = arrays[idx].astype(np.float32, copy=False).ravel()
            selected.append(w)
        if not selected:
            return np.zeros((max_dim,), dtype=np.float32)
        vec = np.concatenate(selected, axis=0)
        if vec.size > max_dim:
            step = math.ceil(vec.size / max_dim)
            vec = vec[::step][:max_dim]
        return vec
    
    embeddings = []
    for dept in dept_names:
        emb = _build_dept_embedding(department_states[dept])
        embeddings.append(emb)
    
    X = np.stack(embeddings, axis=0)  # (n_depts, d)
    
    # Determine optimal number of clusters
    if num_clusters is None:
        actual_max_clusters = min(max_clusters, n_depts - 1)
        num_clusters = determine_optimal_clusters(
            X,
            max_clusters=actual_max_clusters,
            min_clusters=2,
            method="elbow",
            random_state=random_state,
        )
    
    # Run clustering
    num_clusters = max(1, min(num_clusters, n_depts))
    
    if num_clusters >= n_depts:
        # Each department in its own cluster
        clusters = {i: [dept] for i, dept in enumerate(dept_names)}
        dept_to_cluster = {dept: i for i, dept in enumerate(dept_names)}
        return clusters, dept_to_cluster
    
    rng = np.random.RandomState(random_state if random_state is not None else 0)
    labels, _ = _run_kmeans(X, num_clusters, rng, max_iter=100)
    
    # Build cluster mappings
    clusters: Dict[int, List[str]] = defaultdict(list)
    dept_to_cluster: Dict[str, int] = {}
    
    for dept, label in zip(dept_names, labels):
        cid = int(label)
        clusters[cid].append(dept)
        dept_to_cluster[dept] = cid
    
    return dict(clusters), dept_to_cluster

