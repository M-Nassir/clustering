# %% imports
import pandas as pd
from collections import Counter
from sklearn.metrics import adjusted_rand_score

# %%
# TODO: check code is correct
def compute_purity(df, true_col, pred_col):
    """
    Compute purity for a clustering result.

    Parameters:
    - df: DataFrame containing true and predicted labels
    - true_col: Column name for true labels
    - pred_col: Column name for predicted labels

    Returns:
    - purity score (float)
    """
    # Filter out rows with unknown true labels
    valid_df = df[df[true_col] != -1]
    
    total = len(valid_df)
    if total == 0:
        return 0.0
    
    # Group by predicted labels
    grouped = valid_df.groupby(pred_col)[true_col]

    correct = 0
    for _, group in grouped:
        most_common_label = group.mode().iloc[0]
        correct += (group == most_common_label).sum()
    
    return correct / total

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min

# TODO: check code is correct, and why negative homogeneity scores are produced
def compute_homogeneity(df, true_col, pred_col):
    """
    Compute homogeneity for a clustering result.

    Parameters:
    - df: DataFrame containing true and predicted labels
    - true_col: Column name for true labels
    - pred_col: Column name for predicted labels

    Returns:
    - homogeneity score (float)
    """
    # Filter out rows with unknown true labels and predicted labels (i.e., -1 in both columns)
    valid_df = df[(df[true_col] != -1) & (df[pred_col] != -1)]
    
    total = len(valid_df)
    if total == 0:
        return 0.0
    
    # Get the unique clusters and true labels
    clusters = valid_df[pred_col].unique()
    true_labels = valid_df[true_col].unique()
    
    # Calculate the entropy of clusters (H(C))
    cluster_counts = valid_df[pred_col].value_counts(normalize=True)
    H_C = -np.sum(cluster_counts * np.log(cluster_counts))
    
    # Calculate the conditional entropy (H(C|Y))
    H_C_given_Y = 0
    for cluster in clusters:
        cluster_df = valid_df[valid_df[pred_col] == cluster]
        label_counts = cluster_df[true_col].value_counts(normalize=True)
        cluster_entropy = -np.sum(label_counts * np.log(label_counts))
        H_C_given_Y += (len(cluster_df) / total) * cluster_entropy
    
    # Homogeneity: 1 - (H(C|Y) / H(C))
    homogeneity = 1 - (H_C_given_Y / H_C) if H_C != 0 else 1.0

    return homogeneity

# TODO: check code is correct
def compute_ari(df, true_col, pred_col):
    """
    Compute Adjusted Rand Index (ARI) for a clustering result.

    Parameters:
    - df: DataFrame containing true and predicted labels
    - true_col: Column name for true labels
    - pred_col: Column name for predicted labels

    Returns:
    - ARI score (float)
    """
    true_labels = df[true_col]
    predicted_labels = df[pred_col]
    
    return adjusted_rand_score(true_labels, predicted_labels)

from sklearn.metrics import completeness_score

def compute_completeness(df, true_col, pred_col):
    """
    Compute completeness for a clustering result.

    Parameters:
    - df: DataFrame containing true and predicted labels
    - true_col: Column name for true labels
    - pred_col: Column name for predicted labels

    Returns:
    - completeness score (float)
    """
    true_labels = df[true_col]
    predicted_labels = df[pred_col]
    
    return completeness_score(true_labels, predicted_labels)

from sklearn.metrics import v_measure_score

def compute_v_measure(df, true_col, pred_col):
    """
    Compute V-Measure for a clustering result.

    Parameters:
    - df: DataFrame containing true and predicted labels
    - true_col: Column name for true labels
    - pred_col: Column name for predicted labels

    Returns:
    - V-Measure score (float)
    """
    true_labels = df[true_col]
    predicted_labels = df[pred_col]
    
    return v_measure_score(true_labels, predicted_labels)

from sklearn.metrics import normalized_mutual_info_score

def compute_nmi(df, true_col, pred_col):
    """
    Compute Normalized Mutual Information (NMI) for a clustering result.

    Parameters:
    - df: DataFrame containing true and predicted labels
    - true_col: Column name for true labels
    - pred_col: Column name for predicted labels

    Returns:
    - NMI score (float)
    """
    true_labels = df[true_col]
    predicted_labels = df[pred_col]
    
    return normalized_mutual_info_score(true_labels, predicted_labels)

from sklearn.metrics import silhouette_score

def compute_silhouette_score(df, pred_col, features):
    """
    Compute the Silhouette Score for a clustering result without ground truth.

    Parameters:
    - df: DataFrame containing the predicted labels and features
    - pred_col: Column name for predicted labels
    - features: List of feature columns (used to compute similarity)

    Returns:
    - Silhouette score (float)
    """
    features_data = df[features]
    predicted_labels = df[pred_col]
    
    return silhouette_score(features_data, predicted_labels)

from sklearn.metrics import davies_bouldin_score

def compute_davies_bouldin_score(df, pred_col, features):
    """
    Compute the Davies-Bouldin Index for a clustering result without ground truth.

    Parameters:
    - df: DataFrame containing the predicted labels and features
    - pred_col: Column name for predicted labels
    - features: List of feature columns (used to compute similarity)

    Returns:
    - Davies-Bouldin score (float)
    """
    features_data = df[features]
    predicted_labels = df[pred_col]
    
    return davies_bouldin_score(features_data, predicted_labels)

from sklearn.metrics import calinski_harabasz_score

def compute_calinski_harabasz_score(df, pred_col, features):
    """
    Compute the Calinski-Harabasz Index for a clustering result without ground truth.

    Parameters:
    - df: DataFrame containing the predicted labels and features
    - pred_col: Column name for predicted labels
    - features: List of feature columns (used to compute similarity)

    Returns:
    - Calinski-Harabasz score (float)
    """
    features_data = df[features]
    predicted_labels = df[pred_col]
    
    return calinski_harabasz_score(features_data, predicted_labels)
