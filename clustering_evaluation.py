# %%
# ---------------------------- Imports and setup -----------------
import os
import sys
import time
import pandas as pd

# Get the current working directory (where the notebook is running from)
current_dir = os.getcwd()

# Go up 2 levels to reach the 'clustering' project root
root_path = os.path.abspath(os.path.join(current_dir, '../../'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

results_folder = os.path.abspath(os.path.join(os.getcwd(), '../../results'))
    
from clustering.data.synthetic.one_dim_data import generate_clustering_1d_data
from clustering.data.synthetic.one_dim_data_gauss import generate_clustering_1d_gauss_anomalies
from clustering.data.synthetic.two_dim_data_gauss import generate_clustering_2d_gauss_data

from clustering_methods import (
    kmeans_clustering, meanshift_clustering, dbscan_clustering,
    agglomerative_clustering, gmm_clustering, spectral_clustering,
    constrained_kmeans_clustering, copk_means_clustering, hdbscan_clustering, 
    seeded_k_means_clustering, novel_clustering
)

# Plotting tools
from clustering.utilities.plotting import plot_clusters

# Clustering evaluation metrics
from clustering.utilities.evaluation_metrics import (
    compute_purity, compute_homogeneity, compute_ari,
    compute_completeness, compute_v_measure, compute_nmi,
    compute_silhouette_score, compute_davies_bouldin_score,
    compute_calinski_harabasz_score
)

from dec_clustering import run_dec_clustering_from_dataframe

# %%
# ---------------------------- Data Setup ---------------------------------

# Define dataset mode
mode = "1d_gauss"  # Options: "1d_simple", "1d_gauss", "2d_gauss"
k = None  # Number of clusters (used by some algorithms like k-means, we supply ground truth number)

# Load selected dataset and plot
if mode == "1d_simple":
    k = 3
    df = generate_clustering_1d_data(repeat_const=100, percent_labelled=0.03, random_state=None)

elif mode == "1d_gauss":
    k = 3
    df = generate_clustering_1d_gauss_anomalies(random_seed=42,
                                               labelled_percent=0.1,
                                               cluster_params=[(0, 1), (50, 3), (100, 6)],
                                               samples_per_cluster=10000,
                                               include_anomaly_cluster=True,
                                               )

elif mode == "2d_gauss":
    k=5
    df = generate_clustering_2d_gauss_data(n_samples=10000,
                                        n_components=k,
                                        num_features=2,
                                        rand_seed=0,
                                        same_density=False,
                                        labelled_fraction=0.01,
                                        add_anomaly_cluster=True,
                                        plot=True,
                                        )

# Extract feature columns from the DataFrame
feature_columns = [col for col in df.columns if col not in {'y_true', 'y_live'}]

# assign the dataset name
dataset_name = mode

plot_clusters(df, feature_columns, label_column='y_true', 
              title=dataset_name + ' (all data with histogram overlay)', 
              show_seeds_only=False)

# %%
# ---------------------------- Clustering Execution ------------------------

# Flags to enable/disable algorithms
clustering_flags = {
    'KMeans': False,
    'MeanShift': False,
    'DBSCAN': False,
    'HDBSCAN': False,
    'Agglomerative': False,
    'GMM': False,
    'Spectral': False,  # Note: Spectral Clustering may be slow on large datasets
    'ConstrainedKMeans': False,
    'COPKMeans': False,
    'SeededKMeans': True,
    'novel_method': True,

    'DEC': False,
    'S_DEC': False,
    'CDC': False,
}

# Dictionary to record runtimes
runtimes = {}

# Clustering method configuration
clustering_configs = {
    'KMeans': {
        'function': kmeans_clustering,
        'params': {'n_clusters': k, 'target_column': 'y_true'}
    },
    'MeanShift': {
        'function': meanshift_clustering,
        'params': {'target_column': 'y_true_'}
    },
    'DBSCAN': {
        'function': dbscan_clustering,
        'params': {'target_column': 'y_true', 'remap_labels': False}
    },
    'HDBSCAN': {
        'function': hdbscan_clustering,
        'params': {'target_column': 'y_true', 'min_cluster_size': 5, 'min_samples': None, 'remap_labels': False}
    },
    'Agglomerative': {
        'function': agglomerative_clustering,
        'params': {'n_clusters': k, 'target_column': 'y_true'}
    },
    'GMM': {
        'function': gmm_clustering,
        'params': {'n_components': k, 'target_column': 'y_true'}
    },
    'Spectral': {
        'function': spectral_clustering,
        'params': {'n_clusters': k, 'target_column': 'y_true'}
    },
    'ConstrainedKMeans': {
        'function': constrained_kmeans_clustering,
        'params': {'n_clusters': k, 'target_column': 'y_true'}
    },
    'COPKMeans': {
        'function': copk_means_clustering,  # Function for COPKMeans clustering
        'params': {'k': k, 'target_column': 'y_true', 'label_column': 'y_live', 'remap_labels': True}
    },
    'SeededKMeans': {
        'function': seeded_k_means_clustering,
        'params': {'n_clusters': k, 'target_column': 'y_true', 'seeds':'y_live', 'remap_labels': False}       
    },
    'novel_method': {
    'function': novel_clustering,
    'params': {'seeds': 'y_live'}  # Assuming your clustering method uses 'y_live' as seed labels
    },
}

# Run each enabled clustering algorithm
for name, config in clustering_configs.items():
    if clustering_flags.get(name, False):
        print(f"Running {name} with params: {config['params']}") 
        df_c = df.copy()
        start_time = time.time()
        df_c = config['function'](df_c, feature_columns, **config['params'])
        plot_clusters(df_c, feature_columns, label_column=name, title=name)
        df[name] = df_c[name]
        runtimes[name] = time.time() - start_time

# Convert runtimes dict to DataFrame with dataset name
runtime_df = pd.DataFrame([
    {"Algorithm": algo, "Runtime (s)": rt, "Dataset": dataset_name}
    for algo, rt in runtimes.items()
])

# Print the DataFrame
print("\nRuntimes (in seconds):")
print(runtime_df)

# Construct the filename for the runtime results
runtime_filename = os.path.join(results_folder, f"runtime_{dataset_name}.csv")

# Save the runtime DataFrame
runtime_df.to_csv(runtime_filename, index=False)

# %%
# ---------------------------- Supervised Metric Evaluation ------------------------

# Automatically determine enabled clustering methods from flags
clustering_methods = [name for name, enabled in clustering_flags.items() if enabled]

# Define clustering quality metrics requiring ground truth
supervised_metrics = {
    'Homogeneity': compute_homogeneity,
    'ARI': compute_ari,
    'Purity': compute_purity,
    'Completeness': compute_completeness,
    'V-Measure': compute_v_measure,
    'NMI': compute_nmi,
}

# Compute and collect all supervised metrics
supervised_results = []
for metric, func in supervised_metrics.items():
    results = {
        method: func(df, true_col='y_true', pred_col=method)
        for method in clustering_methods
    }
    metric_df = pd.DataFrame.from_dict(results, orient='index', columns=[metric])
    supervised_results.append(metric_df)

# Combine all supervised metrics into one DataFrame
supervised_metrics_df = pd.concat(supervised_results, axis=1)

# Add dataset name as a column
supervised_metrics_df['Dataset'] = dataset_name

print("\nSupervised Clustering Metrics:")
print(supervised_metrics_df)

# Construct the filename for the runtime results
supervised_metrics_filename = os.path.join(results_folder, f"supervised_metrics_{dataset_name}.csv")

# Save the runtime DataFrame
supervised_metrics_df.to_csv(supervised_metrics_filename, index=False)

# %%
# ---------------------------- Unsupervised Metric Evaluation ------------------------

unsupervised_metrics = {
    'Silhouette Score': compute_silhouette_score,
    'Davies-Bouldin Index': compute_davies_bouldin_score,
    'Calinski-Harabasz Index': compute_calinski_harabasz_score,
}

unsupervised_results = []
for metric_name, func in unsupervised_metrics.items():
    results = {}
    for method in clustering_methods:
        try:
            score = func(df, pred_col=method, features=feature_columns)
            results[method] = score
        except Exception as e:
            print(f"Error computing {metric_name} for {method}: {e}")
            results[method] = None
    metric_df = pd.DataFrame.from_dict(results, orient='index', columns=[metric_name])
    unsupervised_results.append(metric_df)

# Combine and display unsupervised metric results
unsupervised_metrics_df = pd.concat(unsupervised_results, axis=1)
# Add dataset name as a column (optional, if you want to include dataset name)
unsupervised_metrics_df['Dataset'] = dataset_name

# Print the unsupervised metrics DataFrame
print("\nUnsupervised Clustering Metrics:")
print(unsupervised_metrics_df)

# Construct the filename for the unsupervised metrics results
unsupervised_metrics_filename = os.path.join(results_folder, f"unsupervised_metrics_{dataset_name}.csv")

# Save the unsupervised metrics DataFrame
unsupervised_metrics_df.to_csv(unsupervised_metrics_filename, index=False)

print(f"\nUnsupervised metrics saved to {unsupervised_metrics_filename}")

# %%
# ---------------------------- DEC clustering method ------------------------

# Copy original DataFrame
df_c = df.copy()

df_dec = run_dec_clustering_from_dataframe(
    df=df_c,
    target_column='y_true',
    n_clusters=10,
    pretrain_epochs=100,
    train_epochs=100,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    save_dir='saves'
)

plot_clusters(df_dec, feature_columns, label_column='cluster', title='DEC clustering', colors=None)

# %%