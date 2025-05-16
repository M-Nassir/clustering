# %% imports
from typing import List
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, completeness_score, v_measure_score,
                             normalized_mutual_info_score, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score,
                             fowlkes_mallows_score, accuracy_score, homogeneity_score
                             )

# ---------------------------- Helper function -----------------------------------
def filter_valid_rows(df: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
    return df[(df[true_col] != -1) & (df[pred_col] != -1)]

# ---------------------------- Supervised metrics -----------------------------------
def compute_accuracy(df, true_col, pred_col):
    valid_df = filter_valid_rows(df, true_col, pred_col)
    if valid_df.empty:
        return 0.0
    return accuracy_score(valid_df[true_col], valid_df[pred_col])

def compute_purity(df, true_col, pred_col):
    valid_df = filter_valid_rows(df, true_col, pred_col)
    total = len(valid_df)
    
    if total == 0:
        return 0.0
    
    grouped = valid_df.groupby(pred_col)[true_col]
    correct = sum((group == group.mode().iloc[0]).sum() for _, group in grouped)
    
    return correct / total

def compute_homogeneity(df, true_col, pred_col):
    valid_df = filter_valid_rows(df, true_col, pred_col)
    if valid_df.empty:
        return 0.0
    return homogeneity_score(valid_df[true_col], valid_df[pred_col])

def compute_completeness(df: pd.DataFrame, true_col: str, pred_col: str) -> float:
    valid_df = filter_valid_rows(df, true_col, pred_col)
    return completeness_score(valid_df[true_col], valid_df[pred_col])

def compute_nmi(df: pd.DataFrame, true_col: str, pred_col: str) -> float:
    valid_df = filter_valid_rows(df, true_col, pred_col)
    return normalized_mutual_info_score(valid_df[true_col], valid_df[pred_col])

def compute_v_measure(df: pd.DataFrame, true_col: str, pred_col: str) -> float:
    valid_df = filter_valid_rows(df, true_col, pred_col)
    return v_measure_score(valid_df[true_col], valid_df[pred_col])

def compute_ari(df: pd.DataFrame, true_col: str, pred_col: str) -> float:
    valid_df = filter_valid_rows(df, true_col, pred_col)
    return adjusted_rand_score(valid_df[true_col], valid_df[pred_col])

def compute_fmi(df, true_col, pred_col):
    valid_df = filter_valid_rows(df, true_col, pred_col)
    return fowlkes_mallows_score(valid_df[true_col], valid_df[pred_col])

# --------------------------- Unsupervised metrics ---------------------------
def compute_silhouette(df: pd.DataFrame, pred_col: str, features: List[str]) -> float:
    valid_df = df[df[pred_col] != -1]
    if valid_df.empty or valid_df[pred_col].nunique() < 2:
        return -1.0  # Convention: not defined for one cluster
    return silhouette_score(valid_df[features], valid_df[pred_col])

def compute_davies_bouldin(df: pd.DataFrame, pred_col: str, features: List[str]) -> float:
    valid_df = df[df[pred_col] != -1]
    if valid_df.empty or valid_df[pred_col].nunique() < 2:
        return -1.0
    return davies_bouldin_score(valid_df[features], valid_df[pred_col])

def compute_calinski_harabasz(df: pd.DataFrame, pred_col: str, features: List[str]) -> float:
    valid_df = df[df[pred_col] != -1]
    if valid_df.empty or valid_df[pred_col].nunique() < 2:
        return -1.0
    return calinski_harabasz_score(valid_df[features], valid_df[pred_col])

