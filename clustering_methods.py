import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from models.cluster import Nassir_clustering
from k_means_constrained import KMeansConstrained
from copkmeans.cop_kmeans import cop_kmeans
import hdbscan

def cluster_with_remapping(df, feature_columns, clusterer, target_column='y_true', remap_labels=False):
    """
    Perform clustering using the specified clustering algorithm and optionally remap labels
    to match the most frequent ground-truth label in each cluster.

    Parameters:
    - df (pd.DataFrame): DataFrame with features and optionally target labels.
    - feature_columns (list): Feature column names.
    - clusterer (sklearn.cluster object): A clustering algorithm with a `fit` method (e.g., KMeans, DBSCAN).
    - target_column (str): Column name for ground-truth labels.
    - remap_labels (bool): Whether to remap the cluster labels to match the most frequent ground-truth label in each cluster.

    Returns:
    - df (pd.DataFrame): DataFrame with predicted cluster labels added as a new column.
    """

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
    """
    Perform KMeans clustering and add a 'KMeans' column to the DataFrame.

    Returns:
    - df (pd.DataFrame): DataFrame with predicted cluster labels in 'KMeans'.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df['KMeans'] = cluster_with_remapping(df, feature_columns, kmeans, target_column, remap_labels)
    return df

def meanshift_clustering(df, feature_columns, target_column='y_true', bandwidth=None, remap_labels=False):
    """
    Perform Mean Shift clustering and add a 'MeanShift' column to the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with features and optionally target labels.
    - feature_columns (list): Feature column names.
    - target_column (str): Column name for ground-truth labels.
    - bandwidth (float or None): Bandwidth for MeanShift. If None, it is estimated.

    Returns:
    - df (pd.DataFrame): DataFrame with 'MeanShift' column of predicted labels.
    """
    bw = bandwidth or estimate_bandwidth(df[feature_columns].to_numpy(), quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bw, bin_seeding=True)
    df['MeanShift'] = cluster_with_remapping(df, feature_columns, ms, target_column, remap_labels)
    return df

def dbscan_clustering(df, feature_columns, target_column='y_true', eps=0.5, min_samples=5, remap_labels=False):
    """
    Perform DBSCAN clustering and add a 'DBSCAN' column to the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with features and optionally target labels.
    - feature_columns (list): Feature column names.
    - target_column (str): Column name for ground-truth labels.
    - eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    - remap_labels (bool): Whether to remap the cluster labels to match the most frequent ground-truth label in each cluster.

    Returns:
    - df (pd.DataFrame): DataFrame with 'DBSCAN' column of predicted labels.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['DBSCAN'] = cluster_with_remapping(df, feature_columns, dbscan, target_column, remap_labels)
    return df

def hdbscan_clustering(df, feature_columns, target_column='y_true', min_cluster_size=5, min_samples=None, remap_labels=False):
    """
    Perform HDBSCAN clustering and add an 'HDBSCAN' column to the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with features and optionally target labels.
    - feature_columns (list): Feature column names.
    - target_column (str): Column name for ground-truth labels.
    - min_cluster_size (int): Minimum size of clusters.
    - min_samples (int or None): Minimum samples per cluster (can be None for default).
    - remap_labels (bool): Whether to remap the cluster labels to match the most frequent ground-truth label in each cluster.

    Returns:
    - df (pd.DataFrame): DataFrame with 'HDBSCAN' column of predicted labels.
    """
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    df['HDBSCAN'] = cluster_with_remapping(df, feature_columns, hdb, target_column, remap_labels)
    return df

def agglomerative_clustering(df, feature_columns, target_column='y_true', n_clusters=3, linkage='ward', remap_labels=False):
    """
    Perform Agglomerative Clustering and add an 'Agglomerative' column to the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with features and optionally target labels.
    - feature_columns (list): Feature column names.
    - target_column (str): Column name for ground-truth labels.
    - n_clusters (int): The number of clusters to form.
    - linkage (str): The linkage criterion to use ('ward', 'complete', 'average', 'single').
    - remap_labels (bool): Whether to remap the cluster labels to match the most frequent ground-truth label in each cluster.

    Returns:
    - df (pd.DataFrame): DataFrame with 'Agglomerative' column of predicted labels.
    """
    agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    df['Agglomerative'] = cluster_with_remapping(df, feature_columns, agglo, target_column, remap_labels)
    return df

def gmm_clustering(df, feature_columns, target_column='y_true', n_components=3, remap_labels=False):
    """
    Perform Gaussian Mixture Model (GMM) clustering and add a 'GMM' column to the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with features and optionally target labels.
    - feature_columns (list): Feature column names.
    - target_column (str): Column name for ground-truth labels.
    - n_components (int): The number of Gaussian components (clusters) to fit.
    - remap_labels (bool): Whether to remap the cluster labels to match the most frequent ground-truth label in each cluster.

    Returns:
    - df (pd.DataFrame): DataFrame with 'GMM' column of predicted labels.
    """
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    df['GMM'] = cluster_with_remapping(df, feature_columns, gmm, target_column, remap_labels)
    return df

def spectral_clustering(df, feature_columns, target_column='y_true', n_clusters=3, affinity='nearest_neighbors', remap_labels=False):
    """
    Perform Spectral Clustering and add a 'Spectral' column to the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with features and optionally target labels.
    - feature_columns (list): Feature column names.
    - target_column (str): Column name for ground-truth labels.
    - n_clusters (int): The number of clusters to form.
    - affinity (str): The method to use to compute the affinity matrix. ('nearest_neighbors', 'rbf', etc.)
    - remap_labels (bool): Whether to remap the cluster labels to match the most frequent ground-truth label in each cluster.

    Returns:
    - df (pd.DataFrame): DataFrame with 'Spectral' column of predicted labels.
    """
    spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=0)
    df['Spectral'] = cluster_with_remapping(df, feature_columns, spectral, target_column, remap_labels)
    return df

def constrained_kmeans_clustering(df, feature_columns, target_column='y_true',
                                  n_clusters=3, size_min=None, size_max=None,
                                  random_state=0, remap_labels=False):
    """
    Perform constrained KMeans clustering using cluster_with_remapping and add a 'ConstrainedKMeans' column to the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with features and optionally target labels.
    - feature_columns (list): Feature column names.
    - target_column (str): Column name for ground-truth labels.
    - n_clusters (int): Number of clusters.
    - size_min (int or None): Minimum cluster size. If None, computed automatically.
    - size_max (int or None): Maximum cluster size. If None, computed automatically.
    - random_state (int): Random seed.
    - remap_labels (bool): Whether to remap cluster labels to dominant true labels.

    Returns:
    - df (pd.DataFrame): DataFrame with predicted labels in 'ConstrainedKMeans'.
    """
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

import numpy as np
from copkmeans.cop_kmeans import cop_kmeans

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
    """
    Perform COPK-means clustering and add a 'COPKMeans' column to the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with features and optionally target labels.
    - feature_columns (list): Feature column names.
    - target_column (str): Column name for ground-truth labels.
    - label_column (str): Column name for available labels to generate constraints.
    - k (int): The number of clusters.
    - remap_labels (bool): Whether to remap the cluster labels to match the most frequent ground-truth label in each cluster.

    Returns:
    - df (pd.DataFrame): DataFrame with predicted cluster labels in 'COPKMeans'.
    """
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

    Parameters:
    - df (pd.DataFrame): DataFrame with features and optionally target labels.
    - feature_columns (list): Feature column names.
    - target_column (str): Column name for ground-truth labels.
    - n_clusters (int): Number of clusters to form.
    - random_state (int): Seed for reproducibility.
    - remap_labels (bool): Whether to remap the cluster labels to match the most frequent ground-truth label in each cluster.

    Returns:
    - df (pd.DataFrame): DataFrame with predicted cluster labels in 'KMeans'.
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

# Kmeans-- (not implemented as need to do it myself, plus dificult to set parameters)
# def kmeans_minus_minus_from_pseudocode(X, k, l, max_iter=100, tol=1e-4, random_state=None):
#     """
#     Direct implementation of k-means-- from pseudocode.

#     Parameters:
#     - X: ndarray of shape (n_samples, n_features), input data
#     - k: int, number of clusters
#     - l: int, number of outliers to remove each iteration
#     - max_iter: int, maximum number of iterations
#     - tol: float, convergence tolerance
#     - random_state: int or None

#     Returns:
#     - C: ndarray of shape (k, n_features), final cluster centers
#     - L: ndarray of shape (l,), indices of outlier points
#     """
#     rng = check_random_state(random_state)
#     n_samples = X.shape[0]

#     # Step 1: Randomly choose k initial centers
#     C = X[rng.choice(n_samples, k, replace=False)]
#     prev_C = np.copy(C)

#     for i in range(max_iter):
#         # Step 4: Compute distances to nearest center
#         distances = np.min(pairwise_distances(X, C), axis=1)

#         # Step 5: Sort distances in descending order
#         sorted_indices = np.argsort(-distances)

#         # Step 6: Select top-l as outliers
#         L = sorted_indices[:l]

#         # Step 7: Remove outliers to get Xi
#         Xi = np.delete(X, L, axis=0)

#         # Step 8-10: Assign inliers to nearest center and compute new means
#         labels = np.argmin(pairwise_distances(Xi, C), axis=1)
#         new_C = np.zeros_like(C)
#         for j in range(k):
#             cluster_points = Xi[labels == j]
#             if len(cluster_points) > 0:
#                 new_C[j] = cluster_points.mean(axis=0)
#             else:
#                 new_C[j] = X[rng.choice(n_samples)]

#         # Step 11: Update centers
#         C = new_C

#         # Step 12: Check convergence
#         if np.linalg.norm(C - prev_C) < tol:
#             break
#         prev_C = np.copy(C)

#     return C, L
