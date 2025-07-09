# %% imports
from typing import List
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, completeness_score, v_measure_score,
                             normalized_mutual_info_score, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score,
                             fowlkes_mallows_score, accuracy_score, homogeneity_score
                             )
import logging
# ---------------------------- Supervised metrics -----------------------------------
def compute_accuracy(df, true_col, pred_col):
    if df.empty:
        return 0.0
    return accuracy_score(df[true_col], df[pred_col])

def compute_purity(df, true_col, pred_col):
    if df.empty:
        return 0.0
    total = len(df)
    grouped = df.groupby(pred_col)[true_col]
    correct = sum((group == group.mode().iloc[0]).sum() for _, group in grouped)
    return correct / total

def compute_homogeneity(df, true_col, pred_col):
    if df.empty:
        return 0.0
    return homogeneity_score(df[true_col], df[pred_col])

def compute_completeness(df, true_col, pred_col):
    if df.empty:
        return 0.0
    return completeness_score(df[true_col], df[pred_col])

def compute_nmi(df, true_col, pred_col):
    if df.empty:
        return 0.0
    return normalized_mutual_info_score(df[true_col], df[pred_col])

def compute_v_measure(df, true_col, pred_col):
    if df.empty:
        return 0.0
    return v_measure_score(df[true_col], df[pred_col])

def compute_ari(df, true_col, pred_col):
    if df.empty:
        return 0.0
    return adjusted_rand_score(df[true_col], df[pred_col])

def compute_fmi(df, true_col, pred_col):
    if df.empty:
        return 0.0
    return fowlkes_mallows_score(df[true_col], df[pred_col])

# --------------------------- Unsupervised metrics ---------------------------
def compute_silhouette(df: pd.DataFrame, pred_col: str, features: List[str]) -> float:
    if df.empty or df[pred_col].nunique() < 2:
        return -1.0
    return silhouette_score(df[features], df[pred_col])

def compute_davies_bouldin(df: pd.DataFrame, pred_col: str, features: List[str]) -> float:
    if df.empty or df[pred_col].nunique() < 2:
        return -1.0
    return davies_bouldin_score(df[features], df[pred_col])

def compute_calinski_harabasz(df: pd.DataFrame, pred_col: str, features: List[str]) -> float:
    if df.empty or df[pred_col].nunique() < 2:
        return -1.0
    return calinski_harabasz_score(df[features], df[pred_col])

def evaluate_clustering_metrics(df, metrics_dict, 
                                dataset_name, clustering_flags, feature_columns):
    """
    Evaluates all clustering metrics (supervised and unsupervised) and saves a unified table.

    Parameters:
    - df (pd.DataFrame): DataFrame with clustering predictions and optionally ground truth in 'y_true'.
    - dataset_name (str): Name of dataset for output naming.
    - clustering_flags (dict): {method_name: bool} for enabled clustering methods.
    - feature_columns (list): Feature columns used for unsupervised metrics.
    """
    logging.info("\n=== Evaluating clustering metrics for dataset: %s ===", dataset_name)
    
    clustering_methods = [name for name, enabled in clustering_flags.items() if enabled]
    results = []

    for method in clustering_methods:
        logging.debug("\n--> Processing method: %s", method)
        row = {'Algorithm': method}

        # Filter for outliers (only once per method)
        df_filtered = df[(df['y_true'] != -1) & (df[method] != -1)]
        logging.debug("Filter out outlier rows for method '%s': %d out of %d remaining", method, len(df_filtered), len(df))

        for metric_name, func, requires_gt in metrics_dict:
            logging.debug("    - Computing metric: %s", metric_name)
            try:
                if requires_gt:
                    if 'y_true' not in df.columns:
                        logging.debug("      Skipping %s: 'y_true' column not found.", metric_name)
                        row[metric_name] = None
                        continue
                    if df_filtered.empty:
                        logging.debug("      Skipping %s: filtered data is empty.", metric_name)
                        row[metric_name] = None
                        continue
                    score = func(df_filtered, true_col='y_true', pred_col=method)
                else:
                    if df_filtered.empty or df_filtered[method].nunique() < 2:
                        logging.debug("      Skipping %s: not enough clusters or empty data.", metric_name)
                        row[metric_name] = None
                        continue
                    score = func(df_filtered, pred_col=method, features=feature_columns)
                
                logging.info("      Score for %s: %.4f", metric_name, score)
                row[metric_name] = score

            except Exception as e:
                logging.debug("      Error computing %s for %s: %s", metric_name, method, e)
                row[metric_name] = None

        results.append(row)

    metrics_df = pd.DataFrame(results).round(4)
    metrics_df['Dataset'] = dataset_name

    logging.info("\n=== Completed evaluation for dataset: %s ===\n", dataset_name)
    return metrics_df


