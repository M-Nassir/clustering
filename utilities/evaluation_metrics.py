# %% imports
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, completeness_score, v_measure_score,
                             normalized_mutual_info_score, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score,
                            fowlkes_mallows_score, accuracy_score,
                             )

# ---------------------------- Supervised metrics -----------------------------------
def compute_accuracy(df, true_col, pred_col):
    y_true = np.asarray(df[true_col])
    y_pred = np.asarray(df[pred_col])

    return accuracy_score(y_true, y_pred)

def compute_purity(df, true_col, pred_col):
    valid_df = df[(df[true_col] != -1) & (df[pred_col] != -1)]
    total = len(valid_df)
    
    if total == 0:
        return 0.0
    
    grouped = valid_df.groupby(pred_col)[true_col]
    correct = sum((group == group.mode().iloc[0]).sum() for _, group in grouped)
    
    return correct / total

def compute_homogeneity(df, true_col, pred_col):
    valid_df = df[(df[true_col] != -1) & (df[pred_col] != -1)]
    total = len(valid_df)
    
    if total == 0:
        return 0.0

    clusters = valid_df[pred_col].unique()
    H_C = -np.sum(valid_df[pred_col].value_counts(normalize=True) * np.log(valid_df[pred_col].value_counts(normalize=True)))
    
    H_C_given_Y = sum(
        (len(cluster_df) / total) * -np.sum(cluster_df[true_col].value_counts(normalize=True) * np.log(cluster_df[true_col].value_counts(normalize=True)))
        for cluster in clusters if (cluster_df := valid_df[valid_df[pred_col] == cluster]).shape[0] > 0
    )

    return 1 - (H_C_given_Y / H_C) if H_C != 0 else 1.0

def compute_completeness(df: pd.DataFrame, true_col: str, pred_col: str) -> float:
    return completeness_score(df[true_col], df[pred_col])

def compute_nmi(df: pd.DataFrame, true_col: str, pred_col: str) -> float:
    return normalized_mutual_info_score(df[true_col], df[pred_col])

# same as nmi but with average method
def compute_v_measure(df: pd.DataFrame, true_col: str, pred_col: str) -> float:
    return v_measure_score(df[true_col], df[pred_col])

def compute_ari(df: pd.DataFrame, true_col: str, pred_col: str) -> float:
    return adjusted_rand_score(df[true_col], df[pred_col])

def compute_fmi(df, true_col, pred_col):
    return fowlkes_mallows_score(df[true_col], df[pred_col])

# --------------------------- Unsupervised metrics ---------------------------
def compute_silhouette(df: pd.DataFrame, pred_col: str, features: List[str]) -> float:

    # Silhouette Score (requires at least 2 clusters)
    if df[pred_col].nunique() < 2:
        return -1.0  # Convention: not defined for one cluster
    return silhouette_score(df[features], df[pred_col])

def compute_davies_bouldin(df: pd.DataFrame, pred_col: str, features: List[str]) -> float:
    return davies_bouldin_score(df[features], df[pred_col])

def compute_calinski_harabasz(df: pd.DataFrame, pred_col: str, features: List[str]) -> float:
    return calinski_harabasz_score(df[features], df[pred_col])
