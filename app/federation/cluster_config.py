"""
Configuration for clustering and FLoRA aggregation.
"""

# Elbow method configuration
ELBOW_MAX_CLUSTERS = 5
ELBOW_MIN_CLUSTERS = 2
ELBOW_METHOD = "elbow"  # or "silhouette"

# FLoRA aggregation configuration
FLORA_MOMENTUM = 0.1  # Momentum for parameter updates (lower = more new info)
FLORA_USE_RESIDUAL = True  # Use residual-based aggregation
FLORA_CLIENT_MOMENTUM = 0.5  # Momentum for client-level mixing

# Department clustering configuration
DEPT_CLUSTERING_ENABLED = True  # Enable department-level clustering
DEPT_MAX_CLUSTERS = 5  # Maximum clusters for departments
DEPT_CLUSTER_MIXING = 0.3  # How much to mix with cluster average (0.0 = no mixing, 1.0 = full mixing)

# Client clustering configuration (within departments)
CLIENT_NUM_CLUSTERS = 1  # Number of clusters for clients within departments (1 = FedAvg)

# General
RANDOM_SEED = 42
MAX_EMBEDDING_DIM = 4096

# Logging
ENABLE_CLUSTER_LOGGING = True
SAVE_CLUSTER_METADATA = True
CLUSTER_METADATA_DIR = "results/cluster_metadata"
