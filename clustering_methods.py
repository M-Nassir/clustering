import numpy as np
import pandas as pd
from scipy.stats import mode

from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from k_means_constrained import KMeansConstrained
from copkmeans.cop_kmeans import cop_kmeans
import hdbscan
from clustering_nassir.cluster import Nassir_clustering

def cluster_with_remapping(df, feature_columns, clusterer, target_column='y_true', remap_labels=False):
    """
    Perform clustering using the specified clustering algorithm and optionally remap cluster labels 
    to match the most frequent ground-truth label in each cluster.

    This function fits the provided clustering model on the features of the DataFrame and, if specified, 
    remaps the generated cluster labels to align with the most frequent target label within each cluster 
    (using the ground-truth labels).

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the features and optionally ground-truth labels.
    - feature_columns (list of str): List of column names to be used as features for clustering.
    - clusterer (sklearn.cluster object or similar): A clustering algorithm that has a `fit` method 
      (e.g., KMeans, DBSCAN, etc.) which will be used to perform clustering.
    - target_column (str, optional): The column name for the ground-truth labels (default is 'y_true'). 
      If provided and `remap_labels` is `True`, the function will attempt to remap the cluster labels 
      to match the most frequent label from the target column within each cluster.
    - remap_labels (bool, optional): If `True`, cluster labels will be remapped to match the most frequent 
      ground-truth label in each cluster. Default is `False`.

    Returns:
    - np.ndarray: An array of predicted cluster labels, possibly remapped according to the target column.

    Raises:
    - ValueError: If `df` is not a DataFrame, if any feature columns are missing from `df`, 
      or if the `target_column` is not found in `df` when `remap_labels` is `True`.
      """

    # Validate input
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")
    
    if not all(col in df.columns for col in feature_columns):
        raise ValueError(f"Some feature columns are missing from the DataFrame: {feature_columns}")
    
    if target_column and target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    features = df[feature_columns].to_numpy()

    # Fit the clustering model
    clusterer.fit(features)
    cluster_labels = clusterer.labels_ if hasattr(clusterer, 'labels_') else clusterer.predict(features)

    # Optionally remap clusters to dominant true labels (if target column exists)
    if remap_labels and target_column in df.columns:
        labels = np.copy(cluster_labels)
        for i in np.unique(cluster_labels):
            if i == -1:  # Skip noise in DBSCAN
                continue
            mask = (cluster_labels == i)
            if np.any(mask):
                labels[mask] = mode(df.loc[mask, target_column])[0]
    else:
        labels = cluster_labels

    return labels

def kmeans_clustering(df, feature_columns, target_column='y_true', n_clusters=3, random_state=0, remap_labels=False):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df['KMeans'] = cluster_with_remapping(df, feature_columns, kmeans, target_column, remap_labels)

    return df

def meanshift_clustering(df, feature_columns, target_column='y_true', bandwidth=None, remap_labels=False):
    bw = bandwidth or estimate_bandwidth(df[feature_columns].to_numpy(), quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bw, bin_seeding=True)
    df['MeanShift'] = cluster_with_remapping(df, feature_columns, ms, target_column, remap_labels)
    return df

def dbscan_clustering(df, feature_columns, target_column='y_true', eps=0.5, min_samples=5, remap_labels=False):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['DBSCAN'] = cluster_with_remapping(df, feature_columns, dbscan, target_column, remap_labels)
    return df

def hdbscan_clustering(df, feature_columns, target_column='y_true', min_cluster_size=5, min_samples=None, remap_labels=False):
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    df['HDBSCAN'] = cluster_with_remapping(df, feature_columns, hdb, target_column, remap_labels)
    return df

def agglomerative_clustering(df, feature_columns, target_column='y_true', n_clusters=3, linkage='ward', remap_labels=False):
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

def generate_constraints_from_labels(df, label_column='y_live'):
    """
    Generate must-link and cannot-link constraints based on the 'y_live' column,
    using all the labels available that are not -1.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the features and labels.
    - label_column (str): The name of the column containing the available labels.
    
    Returns:
    - must_link (list of tuples): List of must-link pairs (indices).
    - cannot_link (list of tuples): List of cannot-link pairs (indices).
    """
    # Get indices of rows with valid labels (not -1)
    valid_indices = df[df[label_column] != -1].index
    
    must_link = []
    cannot_link = []
    
    # Generate must-link and cannot-link constraints
    for i in valid_indices:
        for j in valid_indices:
            if i != j:
                if df[label_column].iloc[i] == df[label_column].iloc[j]:
                    must_link.append((i, j))  # Same label, must-link constraint
                else:
                    cannot_link.append((i, j))  # Different labels, cannot-link constraint

    return must_link, cannot_link

def copk_means_clustering(df, feature_columns, target_column='y_true', label_column='y_live', k=5, remap_labels=False):
    
    # Generate constraints based on the 'y_live' column (excluding -1 labels)
    must_link, cannot_link = generate_constraints_from_labels(df, label_column=label_column)

    # Perform COPK-means clustering
    clusters, centers = cop_kmeans(dataset=df[feature_columns].to_numpy(), k=k, ml=must_link, cl=cannot_link)
    
    # If remapping is required, remap the clusters to match the most frequent ground-truth label
    if remap_labels and target_column in df.columns:
        remapped_labels = np.copy(clusters)
        for cluster_id in np.unique(clusters):
            if cluster_id == -1 or np.sum(clusters == cluster_id) == 0:
                continue

            # Find the most frequent ground-truth label in the cluster
            mask = (clusters == cluster_id)
            most_common_label = mode(df.loc[mask, target_column], keepdims=True).mode[0]
            remapped_labels[mask] = most_common_label
        
        df['COPKMeans'] = remapped_labels
    else:
        # Directly assign the clusters if no remapping is required
        df['COPKMeans'] = clusters
    
    return df

def seeded_k_means_clustering(df, feature_columns, target_column='y_true', seeds='y_live', n_clusters=3, random_state=0, remap_labels=False):
    """
    Perform KMeans clustering with predefined initial centroids calculated from the 'y_live' column
    and add a 'KMeans' column to the DataFrame.
    """

    # Get seed points (where y_live != -1)
    seed_data = df[df[seeds] != -1]

    # Calculate initial centroids from seed points
    if len(seed_data) > 0:
        initial_centroids = seed_data.groupby(seeds)[feature_columns].mean().to_numpy()
    else:
        initial_centroids = None  # No seed data, so we use default KMeans initialization

    # Perform KMeans with the calculated initial centroids
    kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids, n_init=1, random_state=random_state)

    df['SeededKMeans'] = cluster_with_remapping(df, feature_columns, kmeans, target_column, remap_labels)
    return df

# %%
# ---------------------------- Novel clustering method ------------------------

def novel_clustering(df, feature_columns, seeds='y_live'):
    """
    Perform clustering using Nassir_clustering and add a 'Nassir' column to the DataFrame.

    Returns:
    - df (pd.DataFrame): DataFrame with predicted cluster labels in 'Nassir'.
    """

    # Select feature columns and 'y_live' for clustering input
    num_d = df[feature_columns + [seeds]].to_numpy()

    # Instantiate and cluster
    novel_method = Nassir_clustering()
    df['novel_method'] = novel_method.fit(num_d)
    return df
