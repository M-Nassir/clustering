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

import numpy as np
import pandas as pd
import logging

def evaluate_clustering_metrics(
    df,
    metrics_dict,
    dataset_name,
    clustering_flags,
    feature_columns,
    all_results,        # dict {method_name: [labels_run1, labels_run2, ...]}
    y_true_col='y_true',
    outlier_label=-1
):
    """
    Evaluates clustering metrics averaged over multiple runs and returns a table with mean and std.

    Parameters:
    - df (pd.DataFrame): Original DataFrame with features and ground truth 'y_true'.
    - metrics_dict (list): List of (metric_name, func, requires_gt) tuples.
    - dataset_name (str): Dataset name for results tagging.
    - clustering_flags (dict): {method_name: bool} indicating enabled clustering methods.
    - feature_columns (list): Feature column names used for unsupervised metrics.
    - all_results (dict): {method_name: list of cluster label arrays from multiple runs}
    - y_true_col (str): Name of the ground truth column.
    - outlier_label (int): Label used to mark outliers or invalid clusters.

    Returns:
    - pd.DataFrame: Table with columns:
      ['Algorithm', 'Dataset', metric1_mean, metric1_std, metric2_mean, metric2_std, ...]
    """

    logging.info("\n=== Evaluating clustering metrics over multiple runs for dataset: %s ===", dataset_name)

    clustering_methods = [name for name, enabled in clustering_flags.items() if enabled]
    results = []

    for method in clustering_methods:
        if method not in all_results:
            logging.warning(f"Method '{method}' not found in all_results. Skipping.")
            continue
        
        logging.debug(f"\n--> Processing method: {method}")

        metric_scores = {metric_name: [] for metric_name, _, _ in metrics_dict}

        for run_idx, run_labels in enumerate(all_results[method]):
            # Assign cluster labels for this run
            df_run = df.copy()
            df_run[method] = run_labels

            # Filter out outliers in ground truth and predicted clusters
            df_filtered = df_run[
                (df_run[y_true_col] != outlier_label) & (df_run[method] != outlier_label)
            ]

            for metric_name, func, requires_gt in metrics_dict:
                try:
                    if requires_gt:
                        if y_true_col not in df_run.columns:
                            logging.debug(f"Skipping {metric_name}: '{y_true_col}' not found.")
                            score = None
                        elif df_filtered.empty:
                            logging.debug(f"Skipping {metric_name}: filtered data empty for run {run_idx}.")
                            score = None
                        else:
                            score = func(df_filtered, true_col=y_true_col, pred_col=method)
                    else:
                        if df_filtered.empty or df_filtered[method].nunique() < 2:
                            logging.debug(f"Skipping {metric_name}: not enough clusters or empty data in run {run_idx}.")
                            score = None
                        else:
                            score = func(df_filtered, pred_col=method, features=feature_columns)
                except Exception as e:
                    logging.debug(f"Error computing {metric_name} for {method} run {run_idx}: {e}")
                    score = None

                if score is not None:
                    metric_scores[metric_name].append(score)

        # Aggregate metrics: mean and std dev
        row = {'Algorithm': method, 'Dataset': dataset_name}
        for metric_name in metric_scores:
            scores = metric_scores[metric_name]
            if scores:
                row[f"{metric_name}"] = np.mean(scores)
                # row[f"{metric_name}_std"] = np.std(scores)
            else:
                row[f"{metric_name}"] = None
                # row[f"{metric_name}_std"] = None

        results.append(row)

    metrics_df = pd.DataFrame(results)
    logging.info(f"\n=== Completed multi-run evaluation for dataset: {dataset_name} ===\n")
    return metrics_df



