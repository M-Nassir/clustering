#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:47:43 2022
This example will do clustering by a few algorithms on simple spherical
gaussian clusters.

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
import itertools
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
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.clarans import clarans
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn import datasets, metrics

from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans, COPKMeans
from active_semi_clustering.active.pairwise_constraints import ExampleOracle, ExploreConsolidate, MinMax

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

# data
from sklearn.datasets import make_blobs

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
n_samples = 10000
n_components = 8
num_features = 2
std_ = [0.6, 2, 0.2, 0.7,
        3, 0.4, 0.6, 0.6]

# std_ = [0.6, 2, 0.2, 0.7,
#         3, 0.4, 0.6]

X, y_true = make_blobs(
    n_samples=n_samples, centers=n_components, n_features=num_features,
    cluster_std=std_, random_state=0
)

df = pd.DataFrame(X)
df['y_true'] = y_true

# plot only the original clusters
f, axes = plt.subplots(figsize=(12, 6))
axes = sns.scatterplot(x=df[0], y=df[1], hue=df.y_true, palette=colors)

# %% specify additional number of examples to be made into anomalies
percent_labelled_examples = 1/100

# generate indices for data points to be unlabelled
mask = np.random.choice(np.arange(len(df)),
                        replace=False,
                        size=int(len(df) * (percent_labelled_examples)))
df['y_live'] = -1
df.loc[mask, 'y_live'] = df['y_true'][mask].values

number_of_anomalies = np.sum((df['y_live'] == -1).astype(int))
number_of_labelled_data = np.sum((df['y_live'] != -1).astype(int))
p_labelled_data = number_of_labelled_data/df.shape[0]*100

print("number of unlabelled examples: {}".format(number_of_anomalies))
print("number of labelled data: {}".format(number_of_labelled_data))
print("percentage of labelled data: {}%".format(p_labelled_data))

# %% add an additional unknown anomaly cluster

# create the anomaly points, clusters, noise and mislabelled data samples
X2, y_true2 = make_blobs(
    n_samples=300, centers=[(10, 10), (10, 20), (0, 10)], n_features=num_features,
    cluster_std=[8.6, 0.2, 10], random_state=0,
)

# split the data and make all group numbers > 0
y_true2 = np.ones(y_true2.size) * -1
y_live = np.ones(y_true2.size) * -1

# create a dataframe our of the anomaly cluster values and labels
anomaly_points = np.hstack((X2, y_true2.reshape(-1, 1), y_live.reshape(-1, 1)))
temp_anomaly_cluster_df = pd.DataFrame(anomaly_points)
temp_anomaly_cluster_df.columns = df.columns

# join the anomaly cluster to the rest of the data
df2 = pd.concat([df, temp_anomaly_cluster_df], axis=0).reset_index(drop=True)

df2['y_true'] = df2['y_true'].astype(int)

# %% plot data with anomaly cluster
f, axes = plt.subplots(figsize=(12, 6))
axes = sns.scatterplot(x=df2[0], y=df2[1], hue=df2.y_true, palette=colors)

axes.legend(title='Cluster Labels:')

legend = axes.legend_
legend.get_texts()[0].set_text('Anomalies')
axes.set_xlabel('')
axes.set_ylabel('')

if save_switch is True:
    plt.savefig(path_to_save_fig + '2d_gaussian', bbox_inches='tight')

# %% show the labelled seeds only
labelled_examples = df2[df2.y_live != -1]

f, axes = plt.subplots(figsize=(12, 6))
axes = sns.scatterplot(x=labelled_examples[0], y=labelled_examples[1],
                       hue=labelled_examples.y_true, palette=colors)

axes.legend(title='Labelled seeds:')
legend = axes.legend_

axes.set_xlabel('')
axes.set_ylabel('')

if save_switch is True:
    plt.savefig(path_to_save_fig + '2d_gaussian_seeds_only',
                bbox_inches='tight')

# %%
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
#
#                           Nassir Clustering
#
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# remove one of the labelled clusters seeds
# df2.loc[df2['y_true']==6, 'y_live'] = -1

# convert dataframe to numpy array with required columns
features_range = list(range(num_features))
features_range.append('y_live')
num_d = df2[features_range].to_numpy()

start = time.process_time()
nassir = Nassir_clustering()
cluster_labels = nassir.fit(num_d)
print("Nassir clustering execution time: {}".format(time.process_time()-start))

# plot the nassir clustering results
f, axes = plt.subplots(figsize=(12, 6))
axes = sns.scatterplot(x=df2[0], y=df2[1], hue=cluster_labels, palette=colors)

axes.legend(title='Cluster Labels:')
legend = axes.legend_
legend.get_texts()[0].set_text('Anomalies')
axes.set_xlabel('')
axes.set_ylabel('')
# axes.set_yticks([])
# axes.set_title("New Clustering Algorithm")

df2['Nassir'] = cluster_labels

if save_switch is True:
    plt.savefig(path_to_save_fig + '2d_gaussian_nassir_results',
                bbox_inches='tight')

print(classification_report(df2.y_true, df2.Nassir))
print("adjusted_rand_score: {}".
      format(metrics.adjusted_rand_score(df2.y_true, cluster_labels)))
print("accuracy: {}".format(accuracy_score(df2.y_true, cluster_labels)))
print("Adjusted mutual information score: {}".format(
    (metrics.adjusted_mutual_info_score(df2.y_true, cluster_labels))))
print(completeness_score(df2.y_true, cluster_labels))


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
#
#                      Apply clustering algorithms
#
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# %%
do_kmeans = True
do_mini_batch_kmeans = False
do_dbscan = True
do_AP = False
do_mean_shift = False
do_clarans = False
do_gm = False
do_hdbscan = False
do_birch = False
do_semi_supervised_kmeans = False


# # %% semi-supervised clustering (this is not taken further as results similar to k-means)

# # get the input and true labels
# X, y = df2[[0, 1]].values, df2['y_true'].values

# # initalise the list to hold must link contraint tuples
# con_ml = []

# # get all rows where we have a label
# a_temp = df2[df2['y_live'] != -1]

# # enumerate all the must link constraints
# for i in range(8):
#     t1 = a_temp[a_temp['y_live'] == i].index
#     res0 = [(a, b) for idx, a in enumerate(t1) for b in t1[idx + 1:]]
#     con_ml.append(res0)

# flat_ml_con = [item for items in con_ml for item in items]

# # enumerate all the cannat link constraints
# con_cl = []

# # get all rows where we have a label
# a_temp = df2[df2['y_live'] != -1]

# # enumerate all cannot link constraints
# for i in range(8):
#     for j in range(8):

#         if i != j:
#             ti = a_temp[a_temp['y_live'] == i].index
#             tj = a_temp[a_temp['y_live'] == j].index

#             # print(ti)
#             # print(tj)

#             res_t = list(itertools.product(ti, tj))

#             # print(res_t)

#             con_cl.append(res_t)

# flat_cl_con = [item for items in con_cl for item in items]

# # %%
# for i in range(8):
#     for j in range(8):

#         if i != j:

#             # %%
# for i in range(8):
#     print(len(a_temp[a_temp['y_live'] == i]))

# # %% check that no constraints in cl have the same cluster points in sample
# for i in range(1000):
#     v1 = flat_cl_con[i][0]
#     v2 = flat_cl_con[i][1]

#     if df2.iloc[v1]['y_live'] == df2.iloc[v2]['y_live']:
#         print("problem!!!")

# # %% # run the clustering
# start = time.process_time()
# clusterer = COPKMeans(n_clusters=n_components)
# clusterer.fit(X, ml=flat_ml_con, cl=flat_cl_con)
# print("COPKMeans clustering execution time: {}".format(time.process_time()-start))

# # %% plot the clustering results
# cluster_labels = clusterer.labels_

# # plot the PCK-means clustering results
# f, axes = plt.subplots(figsize=(12, 6))
# axes = sns.scatterplot(x=df2[0], y=df2[1], hue=cluster_labels, palette=colors)

# axes.legend(title='Cluster Labels:')
# legend = axes.legend_
# legend.get_texts()[0].set_text('Anomalies')
# axes.set_xlabel('')
# axes.set_ylabel('')
# axes.set_yticks([])
# # axes.set_title("New Clustering Algorithm")

# %% k-means/lloyds algorithms
if do_kmeans is True:
    features_range = list(range(num_features))
    num_d = df2[features_range].to_numpy()

    kmeans = KMeans(n_clusters=n_components, random_state=0).fit(num_d)
    cluster_labels = kmeans.labels_

    # plot the nassir clustering results
    f, axes = plt.subplots(figsize=(12, 6))
    axes.legend(title='Labels')

    #axes.set_title("K-means Clustering")

    axes = sns.scatterplot(x=df2[0], y=df2[1],
                           hue=cluster_labels, palette=colors)

    axes.legend(title='Cluster labels:')

    legend = axes.legend_
    # legend.get_texts()[0].set_text('Cluster 0')
    # legend.get_texts()[1].set_text('Cluster 1')
    # legend.get_texts()[2].set_text('Cluster 2')

    # We change the fontsize of minor ticks label
    axes.tick_params(axis='both', which='major', labelsize=14)
    axes.tick_params(axis='both', which='minor', labelsize=14)

    plt.rcParams['font.size'] = 14

    axes.set_xlabel('')
    axes.set_ylabel('')

    # (perhaps better to use hungarian algorithm)
    labels = np.zeros_like(cluster_labels)
    for i in range(-1, len(np.unique(cluster_labels))):
        mask = (cluster_labels == i)
        labels[mask] = mode(df2.y_true[mask])[0]

    if save_switch is True:
        plt.savefig(path_to_save_fig + '2d_gaussian_kmeans',
                    bbox_inches='tight')

    print(classification_report(df2.y_true, labels))
    print("adjusted_rand_score: {}".
          format(metrics.adjusted_rand_score(df2.y_true, labels)))
    print("accuracy: {}".format(cluster_acc(df2.y_true, labels)[0]))
    print("accuracy: {}".format(accuracy_score(df2.y_true, labels)))

    df2['KMeans'] = labels

# %% DBSCAN
if do_dbscan is True:

    features_range = list(range(num_features))
    num_d = df2[features_range].to_numpy()

    msps = 15
    nearest_neighbors = NearestNeighbors(n_neighbors=msps)
    neighbors = nearest_neighbors.fit(num_d)

    distances, indices = neighbors.kneighbors(num_d)
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

    eps_knee = distances[knee.knee]
    eps_knee = 0.888  # 0.524

    db = DBSCAN(eps=eps_knee, min_samples=msps).fit(num_d)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    cluster_labels = db.labels_

    # plot the nassir clustering results
    f, axes = plt.subplots(figsize=(12, 6))
    axes.legend(title='Labels')

    #axes.set_title("DBSCAN Clustering")
    axes = sns.scatterplot(
        x=df2[0], y=df2[1], hue=cluster_labels, palette=colors)

    axes.set_xlabel('')
    axes.set_ylabel('')

    if save_switch is True:
        plt.savefig(path_to_save_fig + '2d_gaussian_dbscan',
                    bbox_inches='tight')

    print("adjusted_rand_score: {}".
          format(metrics.adjusted_rand_score(df2.y_true, cluster_labels)))

    df2['DBSCAN'] = cluster_labels

    print("accuracy: {}".format(accuracy_score(df2.y_true, cluster_labels)))

# %% HDBSCAN
if do_hdbscan is True:
    features_range = list(range(num_features))
    num_d = df2[features_range].to_numpy()

    hdb = hdbscan.HDBSCAN(min_cluster_size=20).fit(num_d)
    cluster_labels = hdb.labels_

    # plot the nassir clustering results
    f, axes = plt.subplots(figsize=(12, 6))
    axes.legend(title='Labels')
    axes.set_xlabel('')
    axes.set_yticks([])
    axes.set_title("H-DBSCAN Clustering")
    axes = sns.scatterplot(
        x=df2[0], y=df2[1], hue=cluster_labels, palette="tab10")

    print("adjusted_rand_score: {}".
          format(metrics.adjusted_rand_score(df2.y_true, cluster_labels)))

    # (perhaps better to use hungarian algorithm)
    labels = np.zeros_like(cluster_labels)
    for i in range(-1, len(np.unique(cluster_labels))):
        mask = (cluster_labels == i)
        labels[mask] = mode(df2.y_true[mask])[0]

    print("accuracy: {}".format(accuracy_score(df2.y_true, labels)))

    df2['H-DBSCAN'] = cluster_labels


# %% Mean Shift: n^2 slow, requires parameter that is difficult to set
if do_mean_shift is True:
    # convert dataframe to numpy array with required columns
    features_range = list(range(num_features))
    # features_range.append('y_live')

    # labelled_examples = df2[df2.y_live != -1].reset_index()

    # num_d = labelled_examples[features_range].to_numpy()

    labelled_examples = df2.sample(frac=0.1)
    num_d = labelled_examples[features_range].to_numpy()

    # hard to find right bandwidth without ground truth
    bandwidth = estimate_bandwidth(num_d, quantile=0.3)

    mean_shift = MeanShift(bandwidth=None, bin_seeding=True).fit(num_d)
    cluster_labels = mean_shift.labels_

    f, axes = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x=labelled_examples[0], y=labelled_examples[1],
                    hue=cluster_labels, palette='tab10', ax=axes)

    axes.set_xlabel('')
    axes.set_yticks([])
    axes.set_title('MeanShift Clustering')
    axes.legend(title='Labels')

# %% CLARANS (very slow, even with only 1000 points to cluster)
if do_clarans is True:
    features_range = list(range(num_features))
    num_d = df2[features_range].to_numpy()

    # Load data for cluster analysis - 'Lsun' sample.

    # Create BIRCH algorithm to allocate three clusters.
    clarans_instance = clarans(num_d, 5, 6, 4)
    # Run cluster analysis.
    clarans_instance.process()
    # Get allocated clusters.
    clusters = clarans_instance.get_clusters()
    # Visualize obtained clusters.
    visualizer = cluster_visualizer()
    visualizer.append_clusters(clusters, num_d)
    visualizer.show()

    # fig, ax = plt.subplots()
    # plt.scatter(num_d[:,0], num_d[:,1])
    # plt.show()

# %% Birch (very fast, results not as good)
if do_birch is True:
    features_range = list(range(num_features))
    num_d = df2[features_range].to_numpy()

    birch = Birch(n_clusters=8, threshold=0.5).fit(num_d)
    cluster_labels = birch.labels_

    # plot the nassir clustering results
    f, axes = plt.subplots(figsize=(12, 6))
    axes.legend(title='Labels')

    axes.set_xlabel('')
    axes.set_yticks([])
    axes.set_title("Birch Clustering")

    axes = sns.scatterplot(x=df2[0], y=df2[1],
                           hue=cluster_labels, palette='tab10')


# %%
# features_range = list(range(num_features))
# num_d = df2[features_range].to_numpy()
# oracle = ExampleOracle(df2.y_true, max_queries_cnt=100)

# # %%
# active_learner = MinMax(n_clusters=5)
# active_learner.fit(num_d, oracle=oracle)
# pairwise_constraints = active_learner.pairwise_constraints_

# # %% Then, use the constraints to do the clustering.
# clusterer = PCKMeans(n_clusters=5)
# clusterer.fit(num_d, ml=pairwise_constraints[0], cl=pairwise_constraints[1])

# cluster_labels = clusterer.labels_
# # %%
# # plot the nassir clustering results
# f, axes = plt.subplots(figsize=(12, 6))
# axes = sns.scatterplot(x=df2[0], y=df2[1], hue=cluster_labels, palette=colors)

# axes.legend(title='Labels:')
# legend = axes.legend_
# legend.get_texts()[0].set_text('Anomalies')
# legend.get_texts()[1].set_text('Cluster 0')
# legend.get_texts()[2].set_text('Cluster 1')
# legend.get_texts()[3].set_text('Cluster 2')
# legend.get_texts()[4].set_text('Cluster 3')
# #legend.get_texts()[5].set_text('Cluster 4')

# axes.set_xlabel('')
# axes.set_yticks([])

# %% test
# # %% AgglomerativeClustering
# idx = np.random.randint(df2.shape[0], size=int(0.02*df2.shape[0]))
# final_data = df2[[0, 1]].values[idx, :]
# aggloclust = AgglomerativeClustering(n_clusters=20).fit(final_data)

# cluster_labels = aggloclust.labels_

# # plot the nassir clustering results
# f, axes = plt.subplots(figsize=(12, 6))
# axes.legend(title='Labels')

# axes.set_xlabel('')
# axes.set_yticks([])
# axes.set_title("Agglomerative Clustering")

# axes = sns.scatterplot(x=final_data[:, 0], y=final_data[:, 1],
#                        hue=cluster_labels, palette="tab10")


# # %% AgglomerativeClustering
# idx = np.random.randint(df2.shape[0], size=int(0.05*df2.shape[0]))
# final_data = df2[[0, 1]].values[idx, :]
# aggloclust = AgglomerativeClustering(n_clusters=50).fit(final_data)

# # AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
# #             connectivity=None, linkage='ward', memory=None, n_clusters=5)

# cluster_labels = aggloclust.labels_

# # df2['agglomerative'] = cluster_labels

# # plot the nassir clustering results
# f, axes = plt.subplots(figsize=(12, 6))
# axes.legend(title='Labels')

# axes.set_xlabel('')
# axes.set_yticks([])
# axes.set_title("Agglomerative Clustering")


# # customcmap = ListedColormap(["red",
# #                              "blue",
# #                              "green",
# #                              'black',
# #                              'purple', ])

# axes = sns.scatterplot(
#     x=df2[0], y=df2[1], hue=df2.agglomerative, palette="tab10")


# %% STING Clustering
# from pyclustering.cluster.clique import clique
# final_data = df2[[0, 1]].values
# cq = clique(final_data, density_threshold=15, amount_intervals=100)
# cq.process()
# cluster_labels = cq.get_clusters()

# from pyclustering.cluster import cluster_visualizer
# visualizer = cluster_visualizer()
# visualizer.append_clusters(cluster_labels, final_data)
# visualizer.show()

# # %% CURE
# from pyclustering.cluster.cure import cure
# final_data = df2[[0, 1]].values
# cq = cure(final_data, 5)
# cq.process()
# cluster_labels = cq.get_clusters()

# from pyclustering.cluster import cluster_visualizer
# visualizer = cluster_visualizer()
# visualizer.append_clusters(cluster_labels, final_data)
# visualizer.show()

# df2['clique'] = cluster_labels

# # plot the nassir clustering results
# f, axes = plt.subplots(figsize=(12, 6))
# axes.legend(title='Labels')

# axes.set_xlabel('')
# axes.set_yticks([])
# axes.set_title("CLIQUE Clustering")

# axes = sns.scatterplot(x=df2[0], y=df2[1], hue=df2['clique'], palette='tab10')


# %%
# # %% Affinity propagation (very slow)
# clustering = AffinityPropagation(random_state=5).fit(X)

# fig, ax = plt.subplots(figsize=(8, 6))
# sns.scatterplot(x=X[:, 0], y=X[:, 1],
#                 hue=clustering.labels_,
#                 palette="tab10",
#                 legend='brief')
# ax.set_xlabel(r'x', fontsize=14)
# ax.set_ylabel(r'y', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()

# # %% SpectralClustering
# clustering = SpectralClustering().fit(X)
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.scatterplot(x=X[:, 0], y=X[:, 1],
#                 hue=clustering.labels_,
#                 palette="tab10",
#                 legend='brief')
# ax.set_xlabel(r'x', fontsize=14)
# ax.set_ylabel(r'y', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()

# # %% from sklearn.cluster import AgglomerativeClustering
# clustering = AgglomerativeClustering().fit(X)
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.scatterplot(x=X[:, 0], y=X[:, 1],
#                 hue=clustering.labels_,
#                 palette="tab10",
#                 legend='brief')
# ax.set_xlabel(r'x', fontsize=14)
# ax.set_ylabel(r'y', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()

# # %% OPTICS
# optics = OPTICS().fit(X)
# labels = optics.labels_

# fig, ax = plt.subplots(figsize=(8, 6))
# sns.scatterplot(x=X[:, 0], y=X[:, 1],
#                 hue=labels,
#                 palette="tab10",
#                 legend='brief')
# ax.set_xlabel(r'x', fontsize=14)
# ax.set_ylabel(r'y', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()

# # %% Birch
# birch = Birch().fit(X)
# labels = birch.labels_

# fig, ax = plt.subplots(figsize=(8, 6))
# sns.scatterplot(x=X[:, 0], y=X[:, 1],
#                 hue=labels,
#                 palette="tab10",
#                 legend='brief')
# ax.set_xlabel(r'x', fontsize=14)
# ax.set_ylabel(r'y', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()

# # %% Gaussian Mixture Model

# gm = GaussianMixture(n_components=5, random_state=0).fit(X)

# labels = gm.predict(X)

# fig, ax = plt.subplots(figsize=(8, 6))
# sns.scatterplot(x=X[:, 0], y=X[:, 1],
#                 hue=labels,
#                 palette="tab10",
#                 legend='brief')
# ax.set_xlabel(r'x', fontsize=14)
# ax.set_ylabel(r'y', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()
