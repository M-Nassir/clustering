import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from k_means_constrained import KMeansConstrained
from copkmeans.cop_kmeans import cop_kmeans
import hdbscan
from clustering_nassir.cluster import NovelClustering
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from clustpy.deep import DEC
import time
import logging

def remap_clusters_hungarian_with_noise(y_pred, y_true, noise_label=-1):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    # Get all unique labels from both arrays (including noise)
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    # Combine unique labels from both arrays to make sure all are considered
    all_labels = np.unique(np.concatenate([unique_true, unique_pred]))

    # Compute confusion matrix over all samples, using combined label set
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # Hungarian algorithm to find best matching - maximize total matches
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Map predicted labels to true labels (based on matching)
    label_map = {all_labels[col]: all_labels[row] for row, col in zip(row_ind, col_ind)}

    # Remap predicted labels, keep noise_label if not mapped
    remapped = np.full_like(y_pred, fill_value=noise_label)
    for i, label in enumerate(y_pred):
        remapped[i] = label_map.get(label, noise_label)

    return remapped

def cluster_with_remapping(df, feature_columns, clusterer, target_column='y_true', 
                           remap_labels=False):

    # --- Validate input ---
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")
    
    if not all(col in df.columns for col in feature_columns):
        raise ValueError(f"Some feature columns are missing from the DataFrame: {feature_columns}")
    
    if remap_labels and (target_column not in df.columns):
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    features = df[feature_columns].to_numpy()

    # --- Fit the clustering model ---
    clusterer.fit(features)
    cluster_labels = clusterer.labels_ if hasattr(clusterer, 'labels_') else clusterer.predict(features)
    # logging.debug(f"Cluster labels before remapping: {np.unique(cluster_labels)}")
    # logging.debug(f"target column unique values: {np.unique(df[target_column])}")

    # --- Remap labels using Hungarian method if requested ---
    if remap_labels and target_column in df.columns:
        labels = remap_clusters_hungarian_with_noise(cluster_labels, df[target_column].to_numpy())
        # logging.debug(f"Remapped labels: {labels}")
    else:
        labels = cluster_labels

    # logging.debug(f"Cluster labels after remapping: {np.unique(labels)}")
    return labels

def kmeans_clustering(df, feature_columns, target_column='y_true', n_clusters=3, 
                      random_state=0, remap_labels=False):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df['KMeans'] = cluster_with_remapping(df, feature_columns, kmeans, target_column, remap_labels)
    return df

def meanshift_clustering(df, feature_columns, target_column='y_true', bandwidth=None, 
                         remap_labels=False):
    bw = bandwidth or estimate_bandwidth(df[feature_columns].to_numpy(), quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bw, bin_seeding=True)
    df['MeanShift'] = cluster_with_remapping(df, feature_columns, ms, target_column, remap_labels)
    return df

def dbscan_clustering(df, feature_columns, target_column='y_true', eps=0.5, min_samples=5, 
                      remap_labels=False):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['DBSCAN'] = cluster_with_remapping(df, feature_columns, dbscan, target_column, remap_labels)
    return df

def hdbscan_clustering(df, feature_columns, target_column='y_true', min_cluster_size=5, 
                       min_samples=None, remap_labels=False):
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    df['HDBSCAN'] = cluster_with_remapping(df, feature_columns, hdb, target_column, remap_labels)
    return df

def agglomerative_clustering(df, feature_columns, target_column='y_true', n_clusters=None, 
                             linkage='ward', remap_labels=False):
    agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    df['Agglomerative'] = cluster_with_remapping(df, feature_columns, agglo, 
                                                 target_column, remap_labels)
    return df

def gmm_clustering(df, feature_columns, target_column='y_true', n_components=None, remap_labels=False):
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    df['GMM'] = cluster_with_remapping(df, feature_columns, gmm, target_column, remap_labels)
    return df

def spectral_clustering(df, feature_columns, target_column='y_true', n_clusters=None, 
                        affinity='nearest_neighbors', remap_labels=False):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=0)
    df['Spectral'] = cluster_with_remapping(df, feature_columns, spectral, 
                                            target_column, remap_labels)
    return df

# ----------------------------------- Constrained kmeans -----------------------------------
def constrained_kmeans_clustering(df, feature_columns, target_column='y_true',
                                  n_clusters=None, size_min=None, size_max=None,
                                  random_state=0, remap_labels=False):
    features = df[feature_columns].to_numpy()
    n_samples = features.shape[0]

    if size_min is None or size_max is None:
        avg_size = n_samples / n_clusters
        size_min = size_min or max(int(avg_size * 0.5), 1)
        size_max = size_max or int(avg_size * 1.5)

    # Use KMeansConstrained
    clusterer = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        random_state=random_state
    )

    # Use cluster_with_remapping for clustering and label remapping
    labels = cluster_with_remapping(df, feature_columns, clusterer, target_column, remap_labels)
    
    # Add the labels to the DataFrame
    df['ConstrainedKMeans'] = labels
    return df

# ----------------------------------- COPK-means -----------------------------------

from itertools import combinations, product

def generate_constraints_from_labels(df, label_column='y_live'):
    must_link = []
    cannot_link = []
    
    grouped = df[df[label_column] != -1].groupby(label_column)

    for _, group in grouped:
        must_link.extend(combinations(group.index, 2))

    labels = list(grouped.groups.keys())
    for i, l1 in enumerate(labels):
        for l2 in labels[i+1:]:
            cannot_link.extend(product(grouped.groups[l1], grouped.groups[l2]))

    return must_link, cannot_link

def copk_means_clustering(df, feature_columns, target_column='y_true', label_column='y_live', 
                          num_clusters=5, remap_labels=False):
    
    # Generate constraints based on the 'y_live' column (excluding -1 labels)
    must_link, cannot_link = generate_constraints_from_labels(df, label_column=label_column)

    # Perform COPK-means clustering
    clusters, centers = cop_kmeans(dataset=df[feature_columns].to_numpy(), 
                                   k=num_clusters, ml=must_link, cl=cannot_link)
    
    # If remapping is required, remap the clusters to match the most frequent ground-truth label
    if remap_labels and target_column in df.columns:
        remapped_labels = remap_clusters_hungarian_with_noise(clusters, df[target_column].to_numpy())
        df['COPKMeans'] = remapped_labels
    else:
        df['COPKMeans'] = clusters
        
    return df

# ----------------------------------- Seeded kmeans -----------------------------------
def seeded_k_means_clustering(df, feature_columns, target_column='y_true', 
                              seeds='y_live', n_clusters=3, random_state=0, 
                              remap_labels=False):
    """
    Perform KMeans clustering with predefined initial centroids 
    calculated from the 'y_live' column
    and add a 'SeededKMeans' column to the DataFrame.
    """
    # Get seed points (where y_live != -1)
    seed_data = df[df[seeds] != -1]

    if not seed_data.empty:
        grouped = seed_data.groupby(seeds)[feature_columns].mean()
        initial_centroids = grouped.to_numpy()

        if len(initial_centroids) != n_clusters:
            logging.warning(f"Warning: Found {len(initial_centroids)} \
                  seed centroids, but n_clusters={n_clusters}. Falling back to default init.")
            initial_centroids = 'k-means++'
            n_init = 10
        else:
            n_init = 1
    else:
        initial_centroids = 'k-means++'
        n_init = 10

    # Perform KMeans with the calculated initial centroids
    kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids, n_init=n_init, 
                    random_state=random_state)
    df['SeededKMeans'] = cluster_with_remapping(df, feature_columns, kmeans, 
                                                target_column, remap_labels)
    return df

# ----------------------------------- Novel clustering method -----------------------------------
def novel_clustering(df, feature_columns, target_column='y_true', seeds='y_live', remap_labels=False):
    """
    Perform clustering using novel clustering method and add a column to the DataFrame.

    Returns:
    - df (pd.DataFrame): DataFrame with predicted cluster labels.
    """

    # Select feature columns and 'y_live' for clustering input
    num_d = df[feature_columns + [seeds]].to_numpy()

    # Instantiate and cluster
    novel_method = NovelClustering()
    df['novel_method'] = novel_method.fit(num_d)

    return df

def novel_clustering2(df, feature_columns, target_column='y_true', seeds='y_live', remap_labels=False):
    """
    Perform clustering using novel clustering method and add a column to the DataFrame.

    Returns:
    - df (pd.DataFrame): DataFrame with predicted cluster labels.
    """

    # Select feature columns and 'y_live' for clustering input
    num_d = df[feature_columns + [seeds]].to_numpy()

    # Instantiate and cluster
    novel_method2 = NovelClustering2()
    df['novel_method2'] = novel_method2.fit(num_d)
    return df

# ----------------------------------- Deep Embedding Clustering (DEC) ------------------------
def dec_clustering(df, feature_columns, num_clusters=3, 
                   pretrain_epochs=10, clustering_epochs=10, 
                   target_column='y_true', remap_labels=True):
    dec = DEC(n_clusters=num_clusters, pretrain_epochs=pretrain_epochs, 
              clustering_epochs=clustering_epochs)
    dec.fit(df[feature_columns].to_numpy())

    # If remapping is required, remap the clusters to match the most frequent ground-truth label
    if remap_labels and target_column in df.columns:
        remapped_labels = remap_clusters_hungarian_with_noise(dec.labels_, 
                                                              df[target_column].to_numpy())
        df['DEC'] = remapped_labels
    else:
        df['DEC'] = dec.labels_
        
    return df

# ----------------------------------- Run and time clusterings -----------------------------------

def run_and_time_clusterings(df, dataset_name, feature_columns, clustering_configs, clustering_flags, skip_clusterings):
    logging.debug("\n=== Running clustering algorithms for dataset: %s ===", dataset_name)
    
    runtimes = {}
    skip_methods = skip_clusterings.get(dataset_name, set())
    for name, config in clustering_configs.items():
        if clustering_flags.get(name, False):
            if name in skip_methods:
                logging.debug("    Skipping clustering method %s for dataset %s due to skip configuration.", name, dataset_name)
                df[name] = -1  # mark skipped methods with invalid cluster id
                runtimes[name] = None
                continue
            
            logging.debug("\n--> Running clustering method: %s", name)
            logging.debug("    Parameters: %s", config['params'])
            df_c = df.copy()
            start = time.time()
            try:
                result = config['function'](df_c, feature_columns, **config['params'])
                df[name] = result[name]
                elapsed = time.time() - start
                runtimes[name] = elapsed
                logging.debug("    Completed %s in %.4f seconds.", name, elapsed)
            except Exception as e:
                logging.warning("    ERROR while running %s: %s", name, e)
                df[name] = -1  # fallback to invalid cluster id
                runtimes[name] = None

    # Convert runtimes dict to DataFrame with dataset name
    runtime_df = pd.DataFrame([
        {"Algorithm": algo, "Runtime (s)": rt, "Dataset": dataset_name}
        for algo, rt in runtimes.items()
    ])

    logging.debug("\n=== Finished all clusterings for %s ===\n", dataset_name)
    return df, runtime_df

