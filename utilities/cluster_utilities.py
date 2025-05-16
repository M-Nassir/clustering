#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:05:33 2023

@author: nassirmohammad
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment

# specify colours for plotting
colors = {-1: 'red', 0: 'green', 1: 'blue', 2: 'black',
              3: 'orange', 4: 'purple', 5: 'brown',
              6: 'pink', 7: 'cyan', 8: 'darkblue',
              9: 'violet', 10: 'magenta',
          }

def cluster_acc(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    def _make_cost_m(x): return -x + np.max(x)
    indexes = linear_assignment(_make_cost_m(cm))
    indexes = np.concatenate(
        [indexes[0][:, np.newaxis], indexes[1][:, np.newaxis]], axis=-1)
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    acc = np.trace(cm2) / np.sum(cm2)
    return acc

# get the top keywords from a cluster formed by an algorithm


def get_top_keywords(n_terms, pred):
    """This function returns the keywords for each centroid of the KMeans"""
    df = pd.DataFrame(X.todense()).groupby(
        pred).mean()  # groups the TF-IDF vector by cluster
    terms = vectorizer.get_feature_names_out()  # access tf-idf terms
    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        # for each row of the dataframe, find the n terms that have the highest tf idf score
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]]))

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
#         print('\nCluster {}'.format(i))
#         # for each row of the dataframe, find the n terms that have the highest tf idf score
#         print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]]))

def plot_2D(df,
            hue_choice,
            plot_cols,
            cluster_names,
            legend_title='Cluster Labels',
            plot_title="Ground Truth - Iris Data",
            ):

    f, axes = plt.subplots(figsize=(12, 6))
    axes = sns.scatterplot(x=df.iloc[:, plot_cols[0]],
                           y=df.iloc[:, plot_cols[1]],
                           hue=df[hue_choice], palette=colors)

    axes.legend(title=legend_title)
    legend = axes.legend_

    for idx, x in enumerate(cluster_names):
        legend.get_texts()[idx].set_text(x)

    axes.set_title(plot_title)

def get_seeds(df, percent, random_seed=1):

    # generate indices for data points to be unlabelled
    indices = np.random.choice(np.arange(len(df)),
                               replace=False,
                               size=int(len(df)*(1-percent)))

    # add a new column that has some labelled data, and the rest unlabelled
    semi_labeled_y = df['y_true'].copy()
    semi_labeled_y[indices] = -1

    return semi_labeled_y

    # percent_labelled_examples = 0.2

    # # generate indices for data points to be unlabelled
    # np.random.seed(1)
    # indices = np.random.choice(np.arange(len(df)),
    #                            replace=False,
    #                            size=int(
    #                                len(df)*(1-percent_labelled_examples)))

    # # add a new column that has some labelled data, and the rest unlabelled
    # semi_labeled_y = df['y_true'].copy()
    # semi_labeled_y[indices] = -1
    # df['y_live'] = semi_labeled_y

    # print("Length of unlabelled data is (indices): {}".format(len(indices)))
    # print("Length of labelled data is (indices): {}".format(
    #     df.shape[0] - len(indices)))

    # df_seeds = df.loc[(df['y_live'] != -1)]

def load_and_prepare_dataset(dataset_name, label_column='class', percent_labelled=0.05, 
                             min_labels_per_cluster=15):
    """
    Load a dataset from CSV, rename the label column to 'y_true', determine number of clusters,
    and mark a portion of data as labelled, ensuring a minimum number of labelled samples per cluster.

    Parameters:
    - dataset_name (str): Name of the CSV file (relative to "data/tabular/").
    - label_column (str): Column name for true labels in the CSV.
    - percent_labelled (float): Initial fraction of dataset to mark as labelled in 'y_live'.
    - min_labels_per_cluster (int): Minimum number of labelled samples per class.

    Returns:
    - df (pd.DataFrame): Modified DataFrame with 'y_true' and 'y_live' columns.
    - k (int): Number of unique classes (clusters).
    """
    # Read data from CSV
    csv_file_path = f"data/tabular/{dataset_name}"
    df = pd.read_csv(csv_file_path)

    # Ensure label column exists
    if label_column not in df.columns:
        raise ValueError(f"The specified label column '{label_column}' was not found in the dataset.")

    # Rename label column to 'y_true'
    df.rename(columns={label_column: 'y_true'}, inplace=True)

    # Determine number of unique classes
    k = df['y_true'].nunique()

    # Determine required number of labelled samples
    n_min_required = k * min_labels_per_cluster
    n_labelled = max(int(len(df) * percent_labelled), n_min_required)

    # Make sure we don't sample more than available rows
    n_labelled = min(n_labelled, len(df))

    # Sample labelled indices stratified by class
    df['y_live'] = -1

    # Stratified sampling to ensure at least min_labels_per_cluster per class
    labelled_indices = []
    for label in df['y_true'].unique():
        class_indices = df[df['y_true'] == label].index
        n_class_samples = min(len(class_indices), max(min_labels_per_cluster, int(percent_labelled * len(class_indices))))
        labelled_indices.extend(np.random.choice(class_indices, size=n_class_samples, replace=False))

    labelled_indices = np.array(labelled_indices)
    df.loc[labelled_indices, 'y_live'] = df.loc[labelled_indices, 'y_true']

    return df, k
