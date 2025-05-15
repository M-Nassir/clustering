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

def remap_clusters_hungarian_with_noise(y_pred, y_true, noise_label=-1):
    mask = (y_true != noise_label) & (y_pred != noise_label)
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    cm = confusion_matrix(y_true_masked, y_pred_masked)
    row_ind, col_ind = linear_sum_assignment(-cm)
    label_map = {col: row for row, col in zip(row_ind, col_ind)}

    remapped = np.full_like(y_pred, fill_value=noise_label)
    for i, label in enumerate(y_pred):
        if label != noise_label:
            remapped[i] = label_map.get(label, noise_label)

    return remapped

def remap_clusters_hungarian(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    label_map = {col: row for row, col in zip(row_ind, col_ind)}
    return np.vectorize(lambda x: label_map.get(x, x))(y_pred)

def cluster_with_remapping(df, feature_columns, clusterer, target_column='y_true', remap_labels=False):
    """
    Perform clustering using the specified clustering algorithm and optionally remap cluster labels 
    using the Hungarian algorithm to best match ground-truth labels.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the features and optionally ground-truth labels.
    - feature_columns (list of str): List of column names to be used as features for clustering.
    - clusterer (sklearn.cluster object or similar): A clustering algorithm with `fit` method.
    - target_column (str, optional): The ground-truth label column name (default is 'y_true').
    - remap_labels (bool, optional): If `True`, cluster labels will be remapped using Hungarian method 
      for best matching to the true labels.

    Returns:
    - np.ndarray: Array of predicted (possibly remapped) cluster labels.

    Raises:
    - ValueError: If inputs are invalid.
    """
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

    # --- Remap labels using Hungarian method if requested ---
    if remap_labels and target_column in df.columns:
        labels = remap_clusters_hungarian_with_noise(cluster_labels, df[target_column].to_numpy())
        print(f"Remapped labels: {labels}")
    else:
        labels = cluster_labels

    return labels

def kmeans_clustering(df, feature_columns, target_column='y_true', n_clusters=3, 
                      random_state=0, remap_labels=False):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df['KMeans'] = cluster_with_remapping(df, feature_columns, kmeans, target_column, remap_labels)
    return df

def meanshift_clustering(df, feature_columns, target_column='y_true', bandwidth=None, remap_labels=False):
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

def agglomerative_clustering(df, feature_columns, target_column='y_true', n_clusters=3, 
                             linkage='ward', remap_labels=False):
    agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    df['Agglomerative'] = cluster_with_remapping(df, feature_columns, agglo, target_column, remap_labels)
    return df

def gmm_clustering(df, feature_columns, target_column='y_true', n_components=3, remap_labels=False):
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    df['GMM'] = cluster_with_remapping(df, feature_columns, gmm, target_column, remap_labels)
    return df

def spectral_clustering(df, feature_columns, target_column='y_true', n_clusters=3, affinity='nearest_neighbors', remap_labels=False):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=0)
    df['Spectral'] = cluster_with_remapping(df, feature_columns, spectral, target_column, remap_labels)
    return df

def constrained_kmeans_clustering(df, feature_columns, target_column='y_true',
                                  n_clusters=3, size_min=None, size_max=None,
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

# -----------------------------------COPK-means-----------------------------------

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
    clusters, centers = cop_kmeans(dataset=df[feature_columns].to_numpy(), k=num_clusters, ml=must_link, cl=cannot_link)
    
    # If remapping is required, remap the clusters to match the most frequent ground-truth label
    if remap_labels and target_column in df.columns:
        remapped_labels = remap_clusters_hungarian(clusters, df[target_column].to_numpy())
        df['COPKMeans'] = remapped_labels
    else:
        df['COPKMeans'] = clusters
        
    return df

def seeded_k_means_clustering(df, feature_columns, target_column='y_true', seeds='y_live', n_clusters=3, random_state=0, remap_labels=False):
    """
    Perform KMeans clustering with predefined initial centroids calculated from the 'y_live' column
    and add a 'SeededKMeans' column to the DataFrame.
    """
    # Get seed points (where y_live != -1)
    seed_data = df[df[seeds] != -1]

    if not seed_data.empty:
        grouped = seed_data.groupby(seeds)[feature_columns].mean()
        initial_centroids = grouped.to_numpy()

        if len(initial_centroids) != n_clusters:
            print(f"Warning: Found {len(initial_centroids)} seed centroids, but n_clusters={n_clusters}. Falling back to default init.")
            initial_centroids = 'k-means++'
            n_init = 10
        else:
            n_init = 1
    else:
        initial_centroids = 'k-means++'
        n_init = 10

    # Perform KMeans with the calculated initial centroids
    kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids, n_init=n_init, random_state=random_state)
    df['SeededKMeans'] = cluster_with_remapping(df, feature_columns, kmeans, target_column, remap_labels)
    return df

# %%
# ---------------------------- Novel clustering method ------------------------

def novel_clustering(df, feature_columns, seeds='y_live'):
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
