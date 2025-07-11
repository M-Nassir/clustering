# %%
import numpy as np
from clustering_methods import (
    kmeans_clustering, meanshift_clustering, dbscan_clustering,
    agglomerative_clustering, gmm_clustering, spectral_clustering,
    constrained_kmeans_clustering, copk_means_clustering, hdbscan_clustering, 
    seeded_k_means_clustering, novel_clustering, dec_clustering,
    run_metrics_time_clusterings,
)

# Plotting Utilities
from utilities.plotting import (
    plot_clusters, plot_enabled_clusterings, 
    plot_confusion_matrices_for_clustering
)

# Clustering Utilities
from utilities.cluster_utilities import save_df, combine_results, process_df, save_metric_tables_latex
from utilities.evaluation_metrics import (
    compute_accuracy, compute_purity, compute_homogeneity, compute_ari,
    compute_completeness, compute_v_measure, compute_nmi, compute_fmi,
    compute_silhouette, compute_davies_bouldin, compute_calinski_harabasz,
    evaluate_clustering_metrics
)
from utilities.generate_load_data import load_dataset

def make_entry(name, percent_labelled, k, plot_figure=False, standardise=False, random_seed=None):
    return {
        "name": name,
        "percent_labelled": percent_labelled,
        "k": k,
        "plot_figure": plot_figure,
        "standardise": standardise,
        "random_seed": random_seed,
    }

# %% read in dataset; the following datasets have been pre-processed so that the last column
# is the class label, and the rest are features, the class column is integer encoded,
# all feature columns have been given a name. 

dataset_dict = {
    # 0: make_entry("1d_simple", 0.03, 3, plot_figure=False, standardise=False, random_seed=None),
    1: make_entry("1d_gauss", 0.002, 3, plot_figure=False, standardise=False, random_seed=None),
    2: make_entry("2d_gauss", 0.006, 6, plot_figure=True, standardise=False, random_seed=6772), # 4549 6628 743 8858 6772
    3: make_entry("iris", 0.2, 3, plot_figure=False, standardise=False, random_seed=None), # 8338 3480 9093
    4: make_entry("wine", 0.3, 3, plot_figure=False, standardise=False, random_seed=None), # 3169 9942
    5: make_entry("breast_cancer", 0.09, 2, plot_figure=False, standardise=False, random_seed=None), # 1451
    6: make_entry("seeds", 0.2, 3, plot_figure=False, standardise=False, random_seed=None), # 8993
    7: make_entry("glass", 0.3, 6, plot_figure=False, standardise=False, random_seed=None), # 1986
    8: make_entry( "ionosphere_UMAP10", 0.1, 2, plot_figure=False, standardise=False, random_seed=None), # 4574
    9: make_entry(# good example for failure analysis as methods do not perform well
        "yeast", 0.05, 4, plot_figure=False, standardise=False, random_seed=None), 
    10: make_entry(# 21, appears more than 2 clusters, unclear ground truth
        "banknote", 0.02, 2, plot_figure=False, standardise=False, random_seed=None), # 21
    11: make_entry("pendigits", 0.05, 10, plot_figure=False, standardise=False, random_seed=None), # 769 
    12: make_entry("land_mines", 0.3, 5, plot_figure=False, standardise=False, random_seed=None),
    13: make_entry("MNIST_UMAP10", 0.05, 10, plot_figure=False, standardise=False, random_seed=None), # 4470
    14: make_entry("6NewsgroupsUMAP10", 0.02, 10, plot_figure=False, standardise=False, random_seed=None),
    15: make_entry( # highly imbalanced, one class dominates 78%, not good # 6435
        "shuttle", 0.01, 3, plot_figure=False, standardise=False, random_seed=None), #2196 
    16: make_entry("cover_type", percent_labelled=0.01, k=7, plot_figure=False, standardise=False, random_seed=None),
}

# ---------------------------- Clustering Algorithm Setup and Execution ------------------------

# Flags to enable/disable specific clustering algorithms
clustering_flags = {
    # Unsupervised clustering methods
    'KMeans': True,
    'MeanShift': False, 
    'DBSCAN': True,
    'HDBSCAN': True,
    'Agglomerative': True, 
    'GMM': True,
    'Spectral': False, 

    # Semi-supervised clustering methods
    'ConstrainedKMeans': True,
    'COPKMeans': True, 
    'SeededKMeans': True,
    'novel_method': True,

    # Deep learning (self-)unsupervised clustering
    'DEC': True,
}

# Configuration dictionary mapping method names to their functions and parameters
clustering_configs = {
    'KMeans': {
        'function': kmeans_clustering,
        'params': {
            'target_column': 'y_true',
            'remap_labels': True,
        }
    },
    'MeanShift': {
        'function': meanshift_clustering,
        'params': {
            'target_column': 'y_true',
            'remap_labels': True,
        }
    },
    'DBSCAN': {
        'function': dbscan_clustering,
        'params': {
            'target_column': 'y_true',
            'remap_labels': True,
        }
    },
    'HDBSCAN': {
        'function': hdbscan_clustering,
        'params': {
            'target_column': 'y_true',
            'min_cluster_size': 5,
            'min_samples': None,
            'remap_labels': True,
        }
    },
    'Agglomerative': {
        'function': agglomerative_clustering,
        'params': {
            'target_column': 'y_true',
            'remap_labels': True,
        }
    },
    'GMM': {
        'function': gmm_clustering,
        'params': {
            'target_column': 'y_true',
            'remap_labels': True,
        }
    },
    'Spectral': {
        'function': spectral_clustering,
        'params': {
            'target_column': 'y_true',
            'remap_labels': True,
        }
    },
    'ConstrainedKMeans': {
        'function': constrained_kmeans_clustering,
        'params': {
            'target_column': 'y_true',
            'size_min': 15,
            'size_max': None, # set it to df.shape[0]
            'remap_labels': True,
        }
    },
    'COPKMeans': {
        'function': copk_means_clustering,
        'params': {
            'target_column': 'y_true',
            'label_column': 'y_live',
            'remap_labels': True,
        }
    },
    'SeededKMeans': {
        'function': seeded_k_means_clustering,
        'params': {
            'target_column': 'y_true',
            'seeds': 'y_live',
            'remap_labels': True,
        }
    },
    'novel_method': {
        'function': novel_clustering,
        'params': {
            'target_column': 'y_true',
            'seeds': 'y_live',
            'remap_labels': False,  # No label remapping for novel method
        }
    },
    'DEC': {
        'function': dec_clustering,
        'params': {
            'pretrain_epochs': 10,      # Default: 100 (reduced here for quick runs)
            'clustering_epochs': 10,    # Default: 150
            'target_column': 'y_true',
            'remap_labels': True,
        }
    },
}

# ---------------------------- Run Clustering Algorithms and Measure Runtime ------------------------

skip_clustering = { # these methods take too long on these datasets
    "shuttle_trn_with_class": {
        "MeanShift",
        "Agglomerative",
        "Spectral",
        "COPKMeans"
    },
    "cover_type_with_class": {
        "MeanShift",
        "Agglomerative",
        "Spectral",
        "COPKMeans",
        "DBSCAN",
        "ConstrainedKMeans",
    },
}

    # Define all metrics as tuples: (metric_name, metric_function, requires_ground_truth)
selected_metrics = {
    'Accuracy': {'fn': compute_accuracy, 'requires_gt': True},
    'Purity': {'fn': compute_purity, 'requires_gt': True},
    'Homogeneity': {'fn': compute_homogeneity, 'requires_gt': True},
    'Completeness': {'fn': compute_completeness, 'requires_gt': True},
    'V-Measure': {'fn': compute_v_measure, 'requires_gt': True},
    'NMI': {'fn': compute_nmi, 'requires_gt': True},
    'ARI': {'fn': compute_ari, 'requires_gt': True},
    'FMI': {'fn': compute_fmi, 'requires_gt': True},
    
    # Internal metrics (optional to include)
    # 'Silhouette Score': {'fn': compute_silhouette, 'requires_gt': False},
    # 'Davies-Bouldin Index': {'fn': compute_davies_bouldin, 'requires_gt': False},
    # 'Calinski-Harabasz Index': {'fn': compute_calinski_harabasz, 'requires_gt': False},
}
