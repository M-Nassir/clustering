# -*- coding: utf-8 -*-

# %%
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
#
#                                   Setup
#
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# data handling
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap
from bokeh.palettes import d3
import bokeh.models as bmo
import bokeh.plotting as bpl
from bokeh.palettes import Category10_10
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure, show
from sklearn.decomposition import TruncatedSVD
from bokeh.plotting import figure, show, output_notebook
from bokeh.palettes import Spectral10
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
import base64
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt

# from tensorflow.keras.datasets import mnist
from sklearn.cluster import MeanShift
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import plotly_express as px

# utilities
import time
from scipy.stats import mode
from kneed import KneeLocator

# datasets
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_20newsgroups

# algorithms to compare to
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import hdbscan
from cluster import Nassir_clustering

# evaluation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import estimate_bandwidth

# dimension reduction
from sklearn.decomposition import PCA
import umap
import umap.plot

# pre-processing
from nltk.stem import WordNetLemmatizer  # Used to lemmatize words
from nltk.tokenize import word_tokenize  # Used to extract words from documents
from sklearn.feature_extraction.text import TfidfVectorizer

mpl.rcParams['figure.facecolor'] = 'white'

save_switch = False
path_to_save_fig = '/Users/nassirmohammad/Google Drive/docs/A_computational_theory_of_clustering/figures/'

# specify colours for some plots
colors = {-1: 'red', 0: 'green', 1: 'blue', 2: 'black',
              3: 'orange', 4: 'purple', 5: 'brown',
              6: 'pink', 7: 'cyan', 8: 'darkblue',
              9: 'violet', 10: 'magenta', 11: 'black',
          }

# umap metric to use
umap_metric = 'cosine'

# %%---------------------------------------------------------------------
# -----------------------------------------------------------------------
#
#                           Load the data
#
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

load_20news_grp = True
load_mushroom = False
load_penguin = False
load_iris = False

# load 20 newsgroup
# -----------------------------------------------------------------------
if load_20news_grp is True:

    # Selected 6 distinct categories from the 20 newsgroups dataset
    # category names may not necessarily match the target labels

    categories_1 = [
        'soc.religion.christian',
        'comp.graphics',
        'sci.crypt',
        'sci.space',
        'rec.sport.baseball',
        'rec.autos',
        'talk.politics.mideast',
    ]

    categories_paper = [
        'comp.windows.x',
        'rec.motorcycles',
        'rec.sport.hockey',
        'sci.crypt',
        'soc.religion.christian',
        'talk.politics.guns',
    ]

    # fetch the data
    fetched_data = fetch_20newsgroups(subset='all',
                                      categories=categories_paper,
                                      shuffle=True,
                                      remove=('headers', 'footers', 'quotes'))

    y = fetched_data.target

    # get ground truth of clusters, we may not necessarily agree with this!
    ground_k = len(np.unique(y))
    print("The ground truth number of clusters is: {}".format(ground_k))

    # get all the category labels for the fetched data
    category_labels = [fetched_data.target_names[x]
                       for x in fetched_data.target]
    hover_df = pd.DataFrame(category_labels, columns=['category'])

    # show the target labels and corresponding category name
    for idx, cat in enumerate(fetched_data.target_names):
        print(idx, cat)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    for i in range(len(fetched_data.data)):
        word_list = word_tokenize(fetched_data.data[i])
        lemmatized_doc = ""
        for word in word_list:
            lemmatized_doc = lemmatized_doc + \
                " " + lemmatizer.lemmatize(word)
        fetched_data.data[i] = lemmatized_doc

    # tf-id vectorisation
    vectorizer = TfidfVectorizer(
        strip_accents='unicode', stop_words='english', min_df=5)
    X = vectorizer.fit_transform(fetched_data.data)

    print("Shape of tf-idf matrix is {}".format(X.shape))

    # percentage of data to sample as seeds
    p = 2/100

    n_umap_dimensions = 10

# load mushroom data set
# -----------------------------------------------------------------------
if load_mushroom is True:
    pass
# load penguin data set
# -----------------------------------------------------------------------
if load_penguin is True:
    pass
# load iris data set
# -----------------------------------------------------------------------
if load_iris is True:
    pass

# %%---------------------------------------------------------------------
# -----------------------------------------------------------------------
#
#                           Dimension Reduction
#
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# umap embedding and visualisation using seaborn to a number of dimensions
reducer = umap.UMAP(n_components=n_umap_dimensions, metric=umap_metric)
embedding = reducer.fit_transform(X)

# only show for coding, for paper visualisation use embedding to two dimensions
# f, axes = plt.subplots(figsize=(12, 6))
# axes.set_aspect('equal')
# axes = sns.scatterplot(x=embedding[:, 0],
#                        y=embedding[:, 1],
#                        hue=y,
#                        palette=colors)

# plt.title("UMAP plot in 2D of {} dimensional mapping".format(n_umap_dimensions))

# %% embedding to 2D for visualisation
embedding_reducer = umap.UMAP(n_components=2, metric=umap_metric)
embedding_vis = embedding_reducer.fit_transform(X)

# %% plot using umap with labels
f = umap.plot.points(embedding_reducer, labels=y)
plt.gca().set_aspect('equal', 'datalim')

# %% plot using UMAP without labels
f = umap.plot.points(embedding_reducer)
plt.gca().set_aspect('equal', 'datalim')

# %% Plotting UMAP embeddings with custom colors

f, axes = plt.subplots(figsize=(12, 6))
axes = sns.scatterplot(x=embedding_vis[:, 0],
                       y=embedding_vis[:, 1],
                       hue=y,
                       palette=colors,
                       )
axes.legend(title='Cluster Labels:',
            labels=[value for value in categories_paper]
            )

# save the plot
if save_switch is True:
    plt.savefig(path_to_save_fig + '20newsgrp_umap_labelled',
                bbox_inches='tight')

f, axes = plt.subplots(figsize=(12, 6))
axes = sns.scatterplot(x=embedding_vis[:, 0],
                       y=embedding_vis[:, 1],
                       )

# save the plot
if save_switch is True:
    plt.savefig(path_to_save_fig + '20newsgrp_umap_unlabelled',
                bbox_inches='tight')

# %%---------------------------------------------------------------------
# -----------------------------------------------------------------------
#
#                           Sample the labels
#
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# create a dataframe from the embedded data of required dimension and  labels
df = pd.DataFrame(embedding)
df['y_true'] = y

# %% random seed selection (works well if clusters have high probability region)

# get the seeds
# generate indices for data points to be unlabelled
mask = np.random.choice(np.arange(len(df)),
                        replace=False,
                        size=int(len(df) * (p)))
df['y_live'] = -1
df.loc[mask, 'y_live'] = df['y_true'][mask].values

number_of_anomalies = np.sum((df['y_live'] == -1).astype(int))
number_of_labelled_data = np.sum((df['y_live'] != -1).astype(int))
p_labelled_data = np.round(number_of_labelled_data/df.shape[0]*100, 2)

print("number of unlabelled examples: {}".format(number_of_anomalies))
print("number of labelled data: {}".format(number_of_labelled_data))
print("percentage of labelled data: {}%".format(p_labelled_data))
print("Average number of seeds per cluster: {}".format(
    number_of_labelled_data/ground_k))

# code to make one of the clusters unknown
# df.loc[df['y_live'] == 5, 'y_live'] = -1

df_seeds = df.loc[(df['y_live'] != -1)]

f, axes = plt.subplots(figsize=(12, 6))
axes = sns.scatterplot(x=df_seeds[0],
                       y=df_seeds[1],
                       hue=df_seeds['y_true'],
                       palette=colors)
axes.legend(title='Cluster Labels:',
            labels=[value for value in categories_paper]
            )

# %%
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
#
#                           Nassir Clustering
#
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
num_d = np.concatenate([embedding, df['y_live'].values.reshape(-1, 1)], axis=1)
nassir = Nassir_clustering()
cluster_labels = nassir.fit(num_d)
df['Nassir'] = cluster_labels

# remove anomaly from scoring
# cluster_labels = df.loc[np.where(df['Nassir'] != -1)]['Nassir']
# y = df.loc[np.where(df['Nassir'] != -1)]['y_true']

print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, cluster_labels))
print("Completeness: %0.3f" % metrics.completeness_score(y, cluster_labels))
print("V-measure: %0.3f" % metrics.v_measure_score(y, cluster_labels))
print("Adjusted Rand-Index: %.3f" %
      metrics.adjusted_rand_score(y, cluster_labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, cluster_labels, sample_size=1000))

# print("Accuracy: {}".format(cluster_acc(y, cluster_labels)))
print(classification_report(y, cluster_labels))

f, axes = plt.subplots(figsize=(12, 6))
axes = sns.scatterplot(x=df[0],
                       y=df[1],
                       hue=df['Nassir'],
                       palette=colors)
axes.legend(title='Cluster Labels:',
            labels=[value for value in categories_paper]
            )
# %%

category_names = {
    -1: 'anomalies',
    0: 'comp.windows.x',
    1: 'rec.motorcycles',
    2: 'rec.sport.hockey',
    3: 'sci.crypt',
    4: 'soc.religion.christian',
    5: 'talk.politics.guns',
}

f = umap.plot.points(embedding_reducer, labels=df['Nassir'])

f, axes = plt.subplots(figsize=(12, 6))
axes = sns.scatterplot(x=embedding_vis[:, 0],
                       y=embedding_vis[:, 1],
                       hue=df['Nassir'],
                       palette=colors,
                       hue_order=y,
                       )

axes.legend(title='Cluster Labels:',
            labels=[value for value in category_names.values()]
            )

# save the plot
if save_switch is True:
    plt.savefig(path_to_save_fig + '20newsgrp_umap_nassir',
                bbox_inches='tight')

# %% get most common keywords


def get_top_keywords(n_terms, pred):
    """This function returns the keywords for each centroid of the KMeans"""
    df = pd.DataFrame(X.todense()).groupby(
        pred).mean()  # groups the TF-IDF vector by cluster
    terms = vectorizer.get_feature_names_out()  # access tf-idf terms
    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        # for each row of the dataframe, find the n terms that have the highest tf idf score
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]]))

    return terms


df_kw = get_top_keywords(10, cluster_labels)

# %% TODO: add colour to the plot
# load the 20newsgroups dataset
data = fetch_20newsgroups(subset='all',
                          remove=('headers', 'footers', 'quotes'))

# apply TF-IDF vectorization and SVD dimensionality reduction
# vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
# tfidf = vectorizer.fit_transform(data.data)
# svd = TruncatedSVD(n_components=2)
# X = svd.fit_transform(tfidf)

# create a ColumnDataSource object for the plot
source = ColumnDataSource(data=dict(x=embedding_vis[:, 0],
                                    y=embedding_vis[:, 1],
                                    Nassir_label=df['Nassir'],
                                    desc=[doc[:200]
                                          for doc in fetched_data.data]))

# categories = df['Nassir'].unique()
# color_mapping = factor_cmap(
#     'category', palette=Category10[len(categories)], factors=categories)


# create the figure and scatter plot
fig = figure(title="20newsgroups Scatter Plot with Hover",
             tools=[HoverTool(tooltips=[("Description", "@desc")])],
             )

fig.scatter(x='x',
            y='y',
            # color=color_mapping,
            source=source)

# add labels
# labels = LabelSet(x='x', y='y', text='Nassir_label', source=source)
# fig.add_layout(labels)

# display the plot
show(fig)
