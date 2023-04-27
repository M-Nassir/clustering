#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:05:33 2023

@author: nassirmohammad
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment

# specify colours for plotting
colors = {-1: 'red', 0: 'green', 1: 'blue', 2: 'black',
              3: 'orange', 4: 'purple', 5: 'brown',
              6: 'pink', 7: 'cyan', 8: 'darkblue',
              9: 'violet', 10: 'magenta',
          }

# function to match clustering algorithm predictions with ground truth labels
# and return an accuracy score.


def cluster_acc(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    def _make_cost_m(x): return -x + np.max(x)
    indexes = linear_assignment(_make_cost_m(cm))
    indexes = np.concatenate(
        [indexes[0][:, np.newaxis], indexes[1][:, np.newaxis]], axis=-1)
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    acc = np.trace(cm2) / np.sum(cm2)
    return acc, cm2

# TODO: get top keywords


def get_top_keywords(n_terms, pred):
    """This function returns the keywords for each centroid of the KMeans"""
    df = pd.DataFrame(X.todense()).groupby(
        pred).mean()  # groups the TF-IDF vector by cluster
    terms = vectorizer.get_feature_names_out()  # access tf-idf terms
    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        # for each row of the dataframe, find the n terms that have the highest tf idf score
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]]))


# TODO: plot 2D results
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


# TODO: check if required
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
