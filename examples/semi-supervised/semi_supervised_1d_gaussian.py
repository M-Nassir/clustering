#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:47:43 2022

@author: Nassir Mohammad
"""

# %%
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
#
#                               Setup
#
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# clustering algorithms
from coclust.clustering import SphericalKmeans
from sklearn.cluster import KMeans
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
# from pyclustering.cluster import cluster_visualizer
# from pyclustering.cluster.clarans import clarans
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering

# evaluation metrics
from sklearn import metrics
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# data processing
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import plotly_express as px
from bokeh.plotting import show, save, output_notebook, output_file

# utilities
from time import time
import sys
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import estimate_bandwidth
import time
from scipy.stats import mode
from kneed import KneeLocator

# project imports
from models.cluster_utilities import get_seeds, plot_2D, cluster_acc
from cluster import Nassir_clustering

mpl.rcParams['figure.facecolor'] = 'white'

# specify colours for some plots
colors = {-1: 'red', 0: 'green', 1: 'blue', 2: 'black',
              3: 'orange', 4: 'purple', 5: 'brown',
              6: 'pink', 7: 'cyan', 8: 'darkblue',
              9: 'violet', 10: 'magenta',
          }

save_switch = True
path_to_save_fig = '/Users/nassirmohammad/Google Drive/docs/A_computational_theory_of_clustering/figures/'

# %%
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
#
#                         Make the data
#
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# Create the main block of data without anomalies or unknown clusters
data_holder = []

# label number
i = 0

params = [(0, 1), (50, 3), (100, 6)]

# create three gaussian clusters (these will naturally contain anomalies)
for mu, sig in params:

    temp_data = np.expand_dims(np.random.normal(loc=mu, scale=sig,
                                                size=10000), axis=1)

    temp_labels = np.expand_dims(i * np.ones(temp_data.shape[0]), axis=1)
    temp_data_with_label = np.append(temp_data, temp_labels, axis=1)
    data_holder.append(temp_data_with_label)

    # increment label number
    i = i + 1

# make list into array
data_main = np.concatenate(data_holder)

# create the anomalies and mislabelled data samples
data_anomalies_mislablled = np.array([
    [61, 1],
    [58, 1],
    [8.2, -1],
    [8.3, -1],
    [25, -1],
    [40, -1],
    [70, -1],
    [80, -1],
    [95, -1],
    [112, 2],
])

# repeat the normal data to get a larger dataset (optional)
# data_main_repeated = np.repeat(data_main, 10, axis=0)

# add in the additional anomalies
data = np.concatenate((data_main, data_anomalies_mislablled), axis=0)

# shuffle the data row pairs to randomise the order
np.random.shuffle(data)

# make the dataframe with ground truth labels (these may not be 100% accurate)
df = pd.DataFrame(data, columns=['X', 'y_true', ])

# add a little noise to all data points to make them different
noise = np.round(np.random.normal(0, 1, df.shape[0]), 1) * 0.1
df['X'] = df['X'] + noise

# %% sample a small percentage of labelled examples, then
# specify additional number of examples to be made into anomalies
p = 0.5/100

# generate indices for data points to be unlabelled
mask = np.random.choice(np.arange(len(df)),
                        replace=False,
                        size=int(len(df) * (p)))

# add a new column that has some labelled data, and the rest unlabelled
df['y_true'] = df['y_true'].astype(int)
df['y_live'] = -1
df.loc[mask, 'y_live'] = df['y_true'][mask].values

# create an unknown anomaly cluster after the labelling
data_anomaly_cluster = np.array([
    [150, -1, -1],
    [151, -1, -1],
    [152, -1, -1],
    [155, -1, -1],
    [156, -1, -1],
    [157, -1, -1],
    [158, -1, -1],
    [159, -1, -1],
    [151, -1, -1],
    [150, -1, -1],
    [151, -1, -1],
    [152, -1, -1],
    [155, -1, -1],
    [156, -1, -1],
    [157, -1, -1],
    [158, -1, -1],
    [159, -1, -1],
    [151, -1, -1],
])

# make the dataframe and get column names
temp_anomaly_cluster_df = pd.DataFrame(data_anomaly_cluster)
temp_anomaly_cluster_df.columns = df.columns

# join the anomaly cluster to the rest of the data
df2 = pd.concat([df, temp_anomaly_cluster_df], axis=0, ignore_index=True)

number_of_anomalies = np.sum((df2['y_live'] == -1).astype(int))
number_of_labelled_data = np.sum((df2['y_live'] != -1).astype(int))
p_labelled_data = np.round(number_of_labelled_data/df2.shape[0]*100, 2)

print("number of unlabelled examples: {}".format(number_of_anomalies))
print("number of labelled data: {}".format(number_of_labelled_data))
print("percentage of labelled data: {}%".format(p_labelled_data))

# %% plot the data original labelled data, and the semi-labelled data
cols = ['y_true', 'y_live']

for col in cols:

    f, axes = plt.subplots(figsize=(12, 6))
    plt.rcParams['font.size'] = 14

    sns.histplot(df2, x='X', bins=1000, color='lightgrey', ax=axes,)
    plt.xlabel(' ')
    plt.ylabel('Frequency')

    legend_map = {-1: 'Anomalies',
                  0: 'Cluster 1',
                  1: 'Cluster 2',
                  2: 'Cluster 3'}

    sns.scatterplot(x=df2['X'], y=np.zeros_like(df2['X'])+4.9,
                    hue=df2[col], palette=colors,
                    axes=axes,)

    axes.legend(title='Cluster Labels:')

    legend = axes.legend_
    legend.get_texts()[0].set_text('Anomalies')
    legend.get_texts()[1].set_text('Cluster 0')
    legend.get_texts()[2].set_text('Cluster 1')
    legend.get_texts()[3].set_text('Cluster 2')

    axes.set_xlabel('')
    axes.set_yticks([])

    # We change the fontsize of minor ticks label
    axes.tick_params(axis='both', which='major', labelsize=14)
    axes.tick_params(axis='both', which='minor', labelsize=14)

    if save_switch is True:
        plt.savefig(path_to_save_fig + '1d_gaussian_' +
                    col, bbox_inches='tight')

# %%
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
#
#                           Nassir Clustering
#
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# convert dataframe to numpy array with required columns
num_d = df2[['X', 'y_live']].to_numpy()

start_time = time.process_time()
nassir = Nassir_clustering()
cluster_labels = nassir.fit(num_d)
duration = time.process_time()-start_time
print("Nassir clustering execution time: {}".format(duration))

# plot the clustering results
f, axes = plt.subplots(figsize=(12, 6))
sns.histplot(df2, x='X', bins=1000, color='lightgrey', ax=axes,)
plt.xlabel(' ')
plt.ylabel('Frequency')
sns.scatterplot(x=df2['X'], y=np.zeros_like(df2['X'])+6.15,
                hue=cluster_labels, palette=colors, ax=axes)

axes.legend(title='Cluster Labels:')

legend = axes.legend_
legend.get_texts()[0].set_text('Anomalies')
legend.get_texts()[1].set_text('Cluster 0')
legend.get_texts()[2].set_text('Cluster 1')
legend.get_texts()[3].set_text('Cluster 2')

axes.set_xlabel('')
axes.set_yticks([])
# axes.set_title("Nassir's Clustering")

# We change the fontsize of minor ticks label
axes.tick_params(axis='both', which='major', labelsize=14)
axes.tick_params(axis='both', which='minor', labelsize=14)

plt.rcParams['font.size'] = 14

if save_switch is True:
    plt.savefig(path_to_save_fig + '1d_nassir_results', bbox_inches='tight')

df2['Nassir'] = cluster_labels

print(classification_report(df2.y_true, df2.Nassir))
print("adjusted_rand_score: {}".   format(
    metrics.adjusted_rand_score(df2.y_true, cluster_labels)))

# accuracy, conf_matrix = cluster_acc(df2.y_true, df2['Nassir'])
# print("accuracy: {}".format(accuracy))

print("accuracy: {}".format(accuracy_score(df2.y_true, df2['Nassir'])))

# %%
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
#
#                      Apply clustering algorithms
#
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# %% set algorithm runs
do_kmeans = True
do_dbscan = True
do_AP = False
do_mean_shift = False
do_gm = False

# %% k-means/lloyds algorithms
if do_kmeans is True:

    start_time = time.process_time()
    kmeans = KMeans(n_clusters=3, random_state=0).fit(
        df2['X'].values.reshape(-1, 1))
    duration = time.process_time()-start_time
    print("k-means clustering execution time: {}".format(duration))

    cluster_labels = kmeans.labels_

    f, axes = plt.subplots(figsize=(12, 6))
    sns.histplot(df2, x='X', bins=1000, color='lightgrey', ax=axes,)
    plt.xlabel(' ')
    plt.ylabel('Frequency')

    sns.scatterplot(x=df2['X'], y=np.zeros_like(df2['X'])+5.95,
                    hue=cluster_labels, palette=colors, ax=axes)

    # axes.set_xlabel()
    axes.set_yticks([])
    # axes.set_title('KMeans Clustering')
    axes.legend(title='Cluster Labels:')
    legend = axes.legend_
    legend.get_texts()[0].set_text('Cluster 0')
    legend.get_texts()[1].set_text('Cluster 1')
    legend.get_texts()[2].set_text('Cluster 2')

    # We change the fontsize of minor ticks label
    axes.tick_params(axis='both', which='major', labelsize=14)
    axes.tick_params(axis='both', which='minor', labelsize=14)

    plt.rcParams['font.size'] = 14

    # (perhaps better to use hungarian algorithm)
    labels = np.zeros_like(cluster_labels)
    for i in range(-1, len(np.unique(cluster_labels))):
        mask = (cluster_labels == i)
        labels[mask] = mode(df2.y_true[mask])[0]

    if save_switch is True:
        plt.savefig(path_to_save_fig + '1d_kmeans_results',
                    bbox_inches='tight')

    print(classification_report(df2.y_true, labels))
    print("adjusted_rand_score: {}".
          format(metrics.adjusted_rand_score(df2.y_true, labels)))
    print("accuracy: {}".format(cluster_acc(df2.y_true, labels)[0]))
    print("accuracy: {}".format(accuracy_score(df2.y_true, labels)))

    df2['KMeans'] = labels

# %% DBSCAN
if do_dbscan is True:

    msps = 10
    nearest_neighbors = NearestNeighbors(n_neighbors=msps)
    neighbors = nearest_neighbors.fit(df2['X'].values.reshape(-1, 1))

    distances, indices = neighbors.kneighbors(df2['X'].values.reshape(-1, 1))
    distances = np.sort(distances[:, msps-1], axis=0)

    fig = plt.figure(figsize=(5, 5))
    plt.plot(distances)
    plt.xlabel("Points")
    plt.ylabel("Distance")

    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex',
                       direction='increasing', interp_method='polynomial')

    fig = plt.figure(figsize=(5, 5))
    knee.plot_knee()
    plt.xlabel("Points")
    plt.ylabel("Distance")

    print(distances[knee.knee])

    eps_knee = distances[knee.knee]  # not good results
    # eps_knee = 0.049  # manual knee locator using graph
    eps_knee = 0.114

    start_time = time.process_time()
    db = DBSCAN(eps=eps_knee, min_samples=msps).fit(
        df2['X'].values.reshape(-1, 1))
    cluster_labels = db.labels_
    duration = time.process_time()-start_time
    print("DBSCAN clustering execution time: {}".format(duration))

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    f, axes = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x=df2['X'], y=np.zeros_like(df2['X'])+0.15,
                    hue=cluster_labels, palette='tab10', ax=axes)

    axes.set_xlabel('')
    axes.set_yticks([])
    # axes.set_title('DBSCAN Clustering')
    axes.legend(loc="lower left", ncol=4, title='Cluster Labels:')

    # get the labels in right order
    labels = np.zeros_like(cluster_labels)
    for i in range(-1, len(np.unique(cluster_labels))-1):
        # print(i)
        mask = (cluster_labels == i)

        if i > np.max(df2.y_true):
            labels[mask] = i
        else:
            labels[mask] = mode(df2.y_true[mask])[0]

    if save_switch is True:
        plt.savefig(path_to_save_fig + '1d_DBSCAN', bbox_inches='tight')

    print(classification_report(df2.y_true, labels))
    print("adjusted_rand_score: {}".
          format(metrics.adjusted_rand_score(df2.y_true, labels)))
    print(accuracy_score(df2.y_true, labels))

    df2['DBSCAN'] = labels

# %% Affinity propagation (too slow)
if do_AP is True:
    AP = AffinityPropagation(random_state=5).fit(
        df2['X'].values.reshape(-1, 1))
    cluster_labels = AP.labels_

    f, axes = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x=df2['X'], y=np.zeros_like(df2['X'])+0.15,
                    hue=cluster_labels, palette='Spectral', ax=axes)

    axes.set_xlabel('')
    axes.set_yticks([])
    axes.set_title('AffinityPropagation Clustering')
    axes.legend(title='Labels')

    df2['AffinityPropagation'] = cluster_labels

# %% Mean Shift (too slow)
if do_mean_shift is True:
    mean_shift = MeanShift(bandwidth=None).fit(df2['X'].values.reshape(-1, 1))
    cluster_labels = mean_shift.labels_

    f, axes = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x=df2['X'], y=np.zeros_like(df2['X'])+0.15,
                    hue=cluster_labels, palette='Spectral', ax=axes)

    axes.set_xlabel('')
    axes.set_yticks([])
    axes.set_title('MeanShift Clustering')
    axes.legend(title='Labels')

    df2['MeanShift'] = cluster_labels

# %% GaussianMixture
if do_gm is True:
    gm = GaussianMixture(n_components=3, random_state=0).fit(
        df2['X'].values.reshape(-1, 1))

    cluster_labels = gm.predict(df2['X'].values.reshape(-1, 1))

    f, axes = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x=df2['X'], y=np.zeros_like(df2['X'])+0.15,
                    hue=cluster_labels, palette=colors, ax=axes)

    axes.set_xlabel('')
    axes.set_yticks([])
    axes.set_title('GaussianMixture Clustering')
    axes.legend(title='Labels')

    df2['GaussianMixture'] = cluster_labels
