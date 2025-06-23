#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:05:33 2023

@author: nassirmohammad
"""
import os
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import metrics
import logging
# from scipy.optimize import linear_sum_assignment as linear_assignment
# Synthetic data generators    
# from utilities.generate_load_data import (
#     generate_clustering_1d_data, 
#     generate_clustering_1d_gauss_anomalies, 
#     generate_clustering_2d_gauss_data
# )

# # specify colours for plotting
# colors = {-1: 'red', 0: 'green', 1: 'blue', 2: 'black',
#               3: 'orange', 4: 'purple', 5: 'brown',
#               6: 'pink', 7: 'cyan', 8: 'darkblue',
#               9: 'violet', 10: 'magenta',
#           }

# def cluster_acc(y_true, y_pred):
#     cm = metrics.confusion_matrix(y_true, y_pred)
#     def _make_cost_m(x): return -x + np.max(x)
#     indexes = linear_assignment(_make_cost_m(cm))
#     indexes = np.concatenate(
#         [indexes[0][:, np.newaxis], indexes[1][:, np.newaxis]], axis=-1)
#     js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
#     cm2 = cm[:, js]
#     acc = np.trace(cm2) / np.sum(cm2)
#     return acc

# # get the top keywords from a cluster formed by an algorithm


# def get_top_keywords(n_terms, pred):
#     """This function returns the keywords for each centroid of the KMeans"""
#     df = pd.DataFrame(X.todense()).groupby(
#         pred).mean()  # groups the TF-IDF vector by cluster
#     terms = vectorizer.get_feature_names_out()  # access tf-idf terms
#     for i, r in df.iterrows():
#         logging.debug('\nCluster {}'.format(i))
#         # for each row of the dataframe, find the n terms that have the highest tf idf score
#         logging.debug(','.join([terms[t] for t in np.argsort(r)[-n_terms:]]))

# def cluster_acc(y_true, y_pred):
#     cm = metrics.confusion_matrix(y_true, y_pred)
#     def _make_cost_m(x): return -x + np.max(x)
#     indexes = linear_assignment(_make_cost_m(cm))
#     indexes = np.concatenate(
#         [indexes[0][:, np.newaxis], indexes[1][:, np.newaxis]], axis=-1)
#     js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
#     cm2 = cm[:, js]
#     acc = np.trace(cm2) / np.sum(cm2)
#     return acc, cm2

# TODO: get top keywords


# def get_top_keywords(n_terms, pred):
#     """This function returns the keywords for each centroid of the KMeans"""
#     df = pd.DataFrame(X.todense()).groupby(
#         pred).mean()  # groups the TF-IDF vector by cluster
#     terms = vectorizer.get_feature_names_out()  # access tf-idf terms
#     for i, r in df.iterrows():
#         logging.debug('\nCluster {}'.format(i))
#         # for each row of the dataframe, find the n terms that have the highest tf idf score
#         logging.debug(','.join([terms[t] for t in np.argsort(r)[-n_terms:]]))

# def plot_2D(df,
#             hue_choice,
#             plot_cols,
#             cluster_names,
#             legend_title='Cluster Labels',
#             plot_title="Ground Truth - Iris Data",
#             ):

#     f, axes = plt.subplots(figsize=(12, 6))
#     axes = sns.scatterplot(x=df.iloc[:, plot_cols[0]],
#                            y=df.iloc[:, plot_cols[1]],
#                            hue=df[hue_choice], palette=colors)

#     axes.legend(title=legend_title)
#     legend = axes.legend_

#     for idx, x in enumerate(cluster_names):
#         legend.get_texts()[idx].set_text(x)

#     axes.set_title(plot_title)

# def get_seeds(df, percent, random_seed=1):

#     # generate indices for data points to be unlabelled
#     indices = np.random.choice(np.arange(len(df)),
#                                replace=False,
#                                size=int(len(df)*(1-percent)))

#     # add a new column that has some labelled data, and the rest unlabelled
#     semi_labeled_y = df['y_true'].copy()
#     semi_labeled_y[indices] = -1

#     return semi_labeled_y

#     # percent_labelled_examples = 0.2

#     # # generate indices for data points to be unlabelled
#     # np.random.seed(1)
#     # indices = np.random.choice(np.arange(len(df)),
#     #                            replace=False,
#     #                            size=int(
#     #                                len(df)*(1-percent_labelled_examples)))

#     # # add a new column that has some labelled data, and the rest unlabelled
#     # semi_labeled_y = df['y_true'].copy()
#     # semi_labeled_y[indices] = -1
#     # df['y_live'] = semi_labeled_y

#     # logging.debug("Length of unlabelled data is (indices): {}".format(len(indices)))
#     # logging.debug("Length of labelled data is (indices): {}".format(
#     #     df.shape[0] - len(indices)))

#     # df_seeds = df.loc[(df['y_live'] != -1)]



# Define save_df helper
def save_df(df, filename_prefix, dataset_name, results_folder):
    filename = os.path.join(results_folder, f"{filename_prefix}_{dataset_name}.csv")
    df.to_csv(filename, index=False)
    logging.info(f"{filename_prefix.replace('_', ' ').capitalize()} saved to {filename}")

# Define save_metrics helper
# def save_metrics(metrics, prefix, dataset_name, results_folder):
#     """
#     Converts supervised clustering results into a DataFrame, adds metadata, prints and saves it.

#     Parameters:
#     - supervised_results (dict): Dictionary of {method_name: {metric_name: value}}.
#     - dataset_name (str): Name of the dataset used for clustering.
#     - results_folder (str): Folder where results should be saved. Default is 'results'.
#     """
#     # Convert to DataFrame
#     df = pd.DataFrame.from_dict(metrics, orient='index')

#     # Move algorithm names to a column
#     df.reset_index(inplace=True)
#     df.rename(columns={'index': 'Algorithm'}, inplace=True)

#     # Add dataset name
#     df['Dataset'] = dataset_name

#     # Output and save
#     logging.debug(f"\n{prefix} clustering metrics:")
#     print(df)
#     save_df(df, prefix, dataset_name, results_folder=results_folder)

def combine_results(results_folder="results"):
    runtime_files = []
    metrics_files = []

    # Scan folder for files
    for filename in os.listdir(results_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(results_folder, filename)
            # Heuristic: runtime files only have 3 columns
            with open(filepath, 'r') as f:
                header = f.readline()
                num_columns = len(header.strip().split(","))
                if num_columns == 3 and "Runtime" in header:
                    runtime_files.append(filepath)
                elif num_columns > 3:
                    metrics_files.append(filepath)

    # Read and combine runtimes
    runtime_dfs = [pd.read_csv(file) for file in runtime_files]
    all_runtimes = pd.concat(runtime_dfs, ignore_index=True) if runtime_dfs else pd.DataFrame()

    # Read and combine metrics
    metrics_dfs = [pd.read_csv(file) for file in metrics_files]
    all_metrics = pd.concat(metrics_dfs, ignore_index=True) if metrics_dfs else pd.DataFrame()

    if all_metrics.empty:
        print("⚠️ No metrics files found.")
        return pd.DataFrame()  # return empty DataFrame

    if all_runtimes.empty:
        print("⚠️ No runtime files found. The result will only contain metrics.")

    # Merge on Algorithm and Dataset (left join metrics with runtimes)
    combined = pd.merge(all_metrics, all_runtimes, on=["Algorithm", "Dataset"], how="left") if not all_runtimes.empty else all_metrics

    # Optional: sort for readability
    combined = combined.sort_values(by=["Dataset", "Algorithm"]).reset_index(drop=True)

    print("✅ Combined results ready.")
    return combined



