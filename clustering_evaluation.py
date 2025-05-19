
# %%
# ---------------------------- Imports and setup -----------------
import os
import time
import pandas as pd
from IPython.display import display

# Clustering methods
from clustering_methods import (
    kmeans_clustering, meanshift_clustering, dbscan_clustering,
    agglomerative_clustering, gmm_clustering, spectral_clustering,
    constrained_kmeans_clustering, copk_means_clustering, hdbscan_clustering, 
    seeded_k_means_clustering, novel_clustering, dec_clustering
)

# Plotting
from utilities.plotting import plot_clusters
from utilities.cluster_utilities import (load_dataset, 
                                         save_df
)

# Evaluation metrics
from utilities.evaluation_metrics import (
    compute_accuracy, compute_purity, compute_homogeneity, compute_ari,
    compute_completeness, compute_v_measure, compute_nmi,
    compute_fmi,
    compute_silhouette, compute_davies_bouldin,
    compute_calinski_harabasz
)
    
# %%
# ---------------------------- Dataset Configuration ------------------------

# Output directory
results_folder = 'results'
os.makedirs(results_folder, exist_ok=True)

num_clusters = None  
plot_title = None
random_seed = 365 #np.random.randint(0, 10000)

# %% read in dataset
dataset_list = [
    "1d_simple", 
    "1d_gauss", 
    "2d_gauss", 
    "Seed_Data_class.csv"
]

dataset_name = dataset_list[2] 

# Load dataset
#np.random.randint(0, 10000)
df, num_clusters, plot_title = load_dataset(dataset_name, random_seed = 365)
     
# Extract feature columns from the DataFrame
feature_columns = [col for col in df.columns if col not in {'y_true', 'y_live'}]

# %% Plot dataset and seeds only separately
plot_clusters(df, feature_columns, label_column='y_true', 
              title=plot_title, show_seeds_only=False)
plot_clusters(df, feature_columns, label_column='y_live', 
              title=dataset_name + ' (seeds only)', show_seeds_only=True)

# %%
# ---------------------------- Clustering Algorithm Setup ------------------------

# Flags to enable/disable algorithms
clustering_flags = {

    # Unsupervised
    'KMeans': True,
    'MeanShift': True,
    'DBSCAN': True,
    'HDBSCAN': True,
    'Agglomerative': True,
    'GMM': True,
    'Spectral': False,  # Note: Spectral Clustering may be slow on large datasets

    # Semi-supervised 
    'ConstrainedKMeans': True,
    'COPKMeans': True,
    'SeededKMeans': True,
    'novel_method': True,

    # Deep learning (self-)unsupervised 
    'DEC': True,
}

# Clustering method configuration
clustering_configs = {
    'KMeans': {
        'function': kmeans_clustering,
        'params': {'n_clusters': num_clusters, 'target_column': 'y_true', 'remap_labels': True}
    },
    'MeanShift': {
        'function': meanshift_clustering,
        'params': {'target_column': 'y_true', 'remap_labels': True}
    },
    'DBSCAN': {
        'function': dbscan_clustering,
        'params': {'target_column': 'y_true', 'remap_labels': True}
    },
    'HDBSCAN': {
        'function': hdbscan_clustering,
        'params': {'target_column': 'y_true', 'min_cluster_size': 5, 'min_samples': None, 'remap_labels': True}
    },
    'Agglomerative': {
        'function': agglomerative_clustering,
        'params': {'n_clusters': num_clusters, 'target_column': 'y_true', 'remap_labels': True}
    },
    'GMM': {
        'function': gmm_clustering,
        'params': {'n_components': num_clusters, 'target_column': 'y_true', 'remap_labels': True}
    },
    'Spectral': {
        'function': spectral_clustering,
        'params': {'n_clusters': num_clusters, 'target_column': 'y_true', 'remap_labels': True}
    },
    'ConstrainedKMeans': {
        'function': constrained_kmeans_clustering,
        'params': {'n_clusters': num_clusters, 'target_column': 'y_true', 'size_min': 15, 
                   'size_max': df.shape[0], 'remap_labels': True}
    },
    'COPKMeans': {
        'function': copk_means_clustering,  
        'params': {'num_clusters': num_clusters, 'target_column': 'y_true', 'label_column': 'y_live', 'remap_labels': True}
    },
    'SeededKMeans': {
        'function': seeded_k_means_clustering,
        'params': {'n_clusters': num_clusters, 'target_column': 'y_true', 'seeds':'y_live', 'remap_labels': True}       
    },
    'novel_method': {
        'function': novel_clustering,
        'params': {'seeds': 'y_live'}  
    },
    'DEC': {
        'function': dec_clustering,
        'params': {'num_clusters': num_clusters, 'pretrain_epochs':10, 'clustering_epochs':10,
                   'target_column': 'y_true', 'remap_labels': True,}  
        # defaults 100, 150
    },
}

# Apply clustering algorithms
def apply_clustering_algorithms(df, configs, flags, features, plot=True):
    runtimes = {}
    for name, config in configs.items():
        if flags.get(name, False):
            print(f"Running {name} with params: {config['params']}")
            df_c = df.copy()
            start = time.time()
            df[name] = config['function'](df_c, features, **config['params'])[name]
            if plot:
                plot_clusters(df, features, label_column=name, title=name)
            runtimes[name] = time.time() - start
    return df, runtimes

df, runtimes = apply_clustering_algorithms(df, clustering_configs, clustering_flags, 
                                           feature_columns, plot=True)

# Convert runtimes dict to DataFrame with dataset name
runtime_df = pd.DataFrame([
    {"Algorithm": algo, "Runtime (s)": rt, "Dataset": dataset_name}
    for algo, rt in runtimes.items()
])

# Print the runtime and save and display the DataFrame
print("\nRuntimes (in seconds):")
save_df(runtime_df, "runtime", dataset_name, results_folder)
display(runtime_df)

# %%
# ---------------------------- Metric Evaluation ------------------------

# Define all metrics: (metric_name, function, requires_ground_truth)
all_metrics = [
    ('Accuracy', compute_accuracy, True), # same as purity here
    ('Purity', compute_purity, True),
    ('Homogeneity', compute_homogeneity, True),
    ('Completeness', compute_completeness, True),
    ('V-Measure', compute_v_measure, True),# same as NMI here
    ('NMI', compute_nmi, True),
    ('ARI', compute_ari, True),
    ('FMI', compute_fmi, True),
    ('Silhouette Score', compute_silhouette, False),
    ('Davies-Bouldin Index', compute_davies_bouldin, False),
    ('Calinski-Harabasz Index', compute_calinski_harabasz, False),
]
    
def evaluate_clustering_metrics(df, metrics_dict, dataset_name, clustering_flags,feature_columns):
    """
    Evaluates all clustering metrics (supervised and unsupervised) and saves a unified table.

    Parameters:
    - df (pd.DataFrame): DataFrame with clustering predictions and optionally ground truth in 'y_true'.
    - dataset_name (str): Name of dataset for output naming.
    - clustering_flags (dict): {method_name: bool} for enabled clustering methods.
    - feature_columns (list): Feature columns used for unsupervised metrics.
    """
    # Enabled clustering methods
    clustering_methods = [name for name, enabled in clustering_flags.items() if enabled]

    results = []
    for method in clustering_methods:
        row = {'Algorithm': method}
        for metric_name, func, requires_gt in metrics_dict:
            try:
                # compute supervised metrics
                if requires_gt:
                    if 'y_true' not in df.columns:
                        row[metric_name] = None
                        continue
                    score = func(df, true_col='y_true', pred_col=method)
                # compute unsupervised metrics
                else:
                    score = func(df, pred_col=method, features=feature_columns)
                row[metric_name] = score
            except Exception as e:
                print(f"Error computing {metric_name} for {method}: {e}")
                row[metric_name] = None
        results.append(row)

    # Convert to DataFrame
    metrics_df = pd.DataFrame(results).round(4)
    metrics_df['Dataset'] = dataset_name

    return metrics_df

df_metrics = evaluate_clustering_metrics(df=df, metrics_dict=all_metrics, 
                                         dataset_name=dataset_name, 
                                         clustering_flags=clustering_flags, 
                                         feature_columns=feature_columns,  
)

# save the metrics DataFrame and display
save_df(df_metrics, "clustering_metrics", dataset_name, results_folder=results_folder)
display(df_metrics)


