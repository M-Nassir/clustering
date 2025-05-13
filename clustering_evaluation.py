"""
Clustering Evaluation Pipeline
------------------------------
Author: Nassir Mohammad  
Date: 13 May 2025

Description:
This script provides a comprehensive pipeline for evaluating clustering algorithms across a range of datasets 
and scenarios, including unsupervised, semi-supervised, and deep clustering approaches. It supports both fully-labelled 
and partially-labelled data, and is designed to benchmark clustering performance for research purposes.

The pipeline is intended to compare existing methods against a novel clustering algorithm introduced in the research paper:
"A Computational Theory and Semi-Supervised Algorithm for Clustering" by Nassir Mohammad.

Key Features:
-------------
- Supports various data sources:
    - Synthetic 1D/2D data
    - CSV-based tabular data
    - Image datasets
- Enables semi-supervised evaluation by randomly selecting a subset of labelled points.
- Executes multiple clustering algorithms, including:
    - KMeans, DBSCAN, HDBSCAN
    - Seeded KMeans (semi-supervised)
    - Deep Embedded Clustering (DEC)
- Computes both supervised and unsupervised evaluation metrics:
    Supervised:
        - Adjusted Rand Index (ARI)
        - Normalized Mutual Information (NMI)
        - Homogeneity
        - Purity
        - Completeness
        - V-measure
    Unsupervised:
        - Silhouette Score
        - Davies-Bouldin Index
        - Calinski-Harabasz Index
- Plots clustering results showing true vs predicted labels for visual inspection.
- Saves evaluation metrics and runtime statistics to the `results/` directory in CSV format.

Intended Use:
-------------
This pipeline is designed for research and analysis of clustering algorithm performance, especially in 
semi-supervised settings where partial label information is available. It is flexible, modular, and suitable 
for experimentation with new clustering methods.
"""


# %%
# ---------------------------- Imports and setup -----------------
import os
import time
import pandas as pd
import numpy as np

# Synthetic data generators    
from data.synthetic.generate_data import (
    generate_clustering_1d_data, 
    generate_clustering_1d_gauss_anomalies, 
    generate_clustering_2d_gauss_data
)

# Clustering methods
from clustering_methods import (
    kmeans_clustering, meanshift_clustering, dbscan_clustering,
    agglomerative_clustering, gmm_clustering, spectral_clustering,
    constrained_kmeans_clustering, copk_means_clustering, hdbscan_clustering, 
    seeded_k_means_clustering, novel_clustering
)

# deep learning methods
from dec_clustering import run_dec_clustering_from_dataframe

# Plotting
from utilities.plotting import plot_clusters
from utilities.cluster_utilities import load_and_prepare_dataset

# Evaluation metrics
from utilities.evaluation_metrics import (
    compute_purity, compute_homogeneity, compute_ari,
    compute_completeness, compute_v_measure, compute_nmi,
    compute_fmi,
    compute_silhouette, compute_davies_bouldin,
    compute_calinski_harabasz
)

# Output directory
results_folder = 'results'
os.makedirs(results_folder, exist_ok=True)

# Define save_df helper
def save_df(df, filename_prefix, dataset_name):
    filename = os.path.join(results_folder, f"{filename_prefix}_{dataset_name}.csv")
    df.to_csv(filename, index=False)
    print(f"{filename_prefix.replace('_', ' ').capitalize()} saved to {filename}")
        
# %%
# ---------------------------- Dataset Configuration ------------------------

# Define dataset name, note all features must be numeric
dataset_name = "2d_gauss"  # Options: "1d_simple", 
#                              "1d_gauss", 
#                              "2d_gauss", 
#                              "Seed_Data_class.csv" 

num_clusters = None  
plot_title = None
random_seed = np.random.randint(0, 10000)
gauss_feature_numbers = 2 

# %% read in dataset
if dataset_name == "1d_simple":
    num_clusters = 3
    df = generate_clustering_1d_data(repeat_const=100, 
                                     percent_labelled=0.03, 
                                     random_state=random_seed)
    plot_title = dataset_name + ' (all data with histogram overlay)'

elif dataset_name == "1d_gauss":
    num_clusters = 3
    df = generate_clustering_1d_gauss_anomalies(random_seed=random_seed,
                                               labelled_percent=0.1,
                                               cluster_params=[
                                                   (0, 1), (50, 3), (100, 6)
                                                   ],
                                               samples_per_cluster=10000,
                                               include_anomaly_cluster=True,
                                               )
    plot_title = dataset_name + ' (all data with histogram overlay)'

# TODO: improve the 2d generation of clusters function, e.g. pass std_dev
elif dataset_name == "2d_gauss":
    num_clusters=5
    
    # Define cluster standard deviations
    same_density = False
    if same_density:
        std_dev = 0.6
    else:
        # Set different std deviations for each component
        std_dev = [1.5, 0.8, 1.2, 3, 0.4][:num_clusters]
        
    df = generate_clustering_2d_gauss_data(n_samples=10000,
                                        n_components=num_clusters,
                                        num_features=gauss_feature_numbers,
                                        rand_seed=random_seed,
                                        same_density=False,
                                        labelled_fraction=0.01,
                                        add_anomaly_cluster=True,
                                        std_dev=std_dev,
                                        )
    plot_title = dataset_name + ' (all data)'

else:
    df, num_clusters = load_and_prepare_dataset(dataset_name, 
                                                label_column='class', 
                                                percent_labelled=0.05
                                                )
     
# Extract feature columns from the DataFrame
feature_columns = [col for col in df.columns if col not in {'y_true', 'y_live'}]

# %% Plot dataset and seeds only separately
plot_clusters(df, feature_columns, label_column='y_true', title=plot_title, show_seeds_only=False)
plot_clusters(df, feature_columns, label_column='y_live', title=dataset_name + ' (seeds only)', show_seeds_only=True)

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

    # Deep learning unsupervised and semi-supervised
    'DEC': False,
    'S_DEC': False,
    'CDC': False,
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
}

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

# Print the runtime and DataFrame
print("\nRuntimes (in seconds):")
print(runtime_df)

save_df(runtime_df, "runtime", dataset_name)

# %%
# ---------------------------- Supervised Evaluation ------------------------

# Automatically determine enabled clustering methods from flags
clustering_methods = [name for name, enabled in clustering_flags.items() if enabled]

# Define clustering quality metrics requiring ground truth
supervised_metrics = {
    'Purity': compute_purity,
    'Homogeneity': compute_homogeneity,
    'Completeness': compute_completeness,
    'V-Measure': compute_v_measure,
    'NMI': compute_nmi,
    'ARI': compute_ari,
    'FMI': compute_fmi,
}

# Compute all metrics in one nested dictionary: {algorithm: {metric: value}}
supervised_results = {
    method: {
        metric: func(df, true_col='y_true', pred_col=method)
        for metric, func in supervised_metrics.items()
    }
    for method in clustering_methods
}

# Convert to DataFrame
supervised_metrics_df = pd.DataFrame.from_dict(supervised_results, orient='index')

# Move algorithm names to a column
supervised_metrics_df.reset_index(inplace=True)
supervised_metrics_df.rename(columns={'index': 'Algorithm'}, inplace=True)

# Add dataset name
supervised_metrics_df['Dataset'] = dataset_name

# Output and save
print("\nSupervised Clustering Metrics:")
print(supervised_metrics_df)

save_df(supervised_metrics_df, "supervised_metrics", dataset_name)

# %% ---------------------------- Unsupervised Evaluation ------------------------

unsupervised_metrics = {
    'Silhouette Score': compute_silhouette,
    'Davies-Bouldin Index': compute_davies_bouldin,
    'Calinski-Harabasz Index': compute_calinski_harabasz,
}

# Compute all unsupervised metrics in a nested dictionary: {algorithm: {metric: value}}
unsupervised_results = {}
for method in clustering_methods:
    method_results = {}
    for metric_name, func in unsupervised_metrics.items():
        try:
            score = func(df, pred_col=method, features=feature_columns)
            method_results[metric_name] = score
        except Exception as e:
            print(f"Error computing {metric_name} for {method}: {e}")
            method_results[metric_name] = None
    unsupervised_results[method] = method_results

# Convert to DataFrame
unsupervised_metrics_df = pd.DataFrame.from_dict(unsupervised_results, orient='index')

# Move algorithm names to a column
unsupervised_metrics_df.reset_index(inplace=True)
unsupervised_metrics_df.rename(columns={'index': 'Algorithm'}, inplace=True)

# Add dataset name
unsupervised_metrics_df['Dataset'] = dataset_name

# Output and save
print("\nUnsupervised Clustering Metrics:")
print(unsupervised_metrics_df)

save_df(unsupervised_metrics_df, "unsupervised_metrics", dataset_name)

# %%
# ---------------------------- DEC clustering method ------------------------

df_dec = run_dec_clustering_from_dataframe(
    df.copy(),
    target_column='y_true',
    n_clusters=num_clusters,
    pretrain_epochs=100,
    train_epochs=100,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    save_dir='saves'
)

plot_clusters(df_dec, feature_columns, label_column='cluster', title='DEC clustering', colors=None)