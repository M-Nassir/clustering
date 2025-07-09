# -*- coding: utf-8 -*-

# %%
# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------

import os
import numpy as np
import pandas as pd

import sklearn.datasets as sk_data
from sklearn.datasets import fetch_20newsgroups

import nltk
from nltk.stem import WordNetLemmatizer  
from nltk.tokenize import word_tokenize  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from wordcloud import STOPWORDS
from wordcloud import WordCloud
from sklearn.preprocessing import Normalizer

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import plotly_express as px

from sklearn.decomposition import PCA
import umap
import umap.plot

# %% Paths
# Get project root (assumes script is in a subfolder like 'utilities')
CURRENT_DIR = os.getcwd()
ROOT_PATH = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
DATA_DIR = os.path.join(ROOT_PATH, "data", 'raw', "tabular")
SAVE_DATA_DIR = os.path.join(ROOT_PATH, "data", 'processed')

mpl.rcParams['figure.facecolor'] = 'white'

# Colour palette for plots - use seaborn colour palette instead for consistency
colors = sns.color_palette("tab10", 11)
color_map = {i: colors[i] for i in range(11)}

save_switch = False
path_to_save_fig = os.path.expanduser('~/Google Drive/docs/A_computational_theory_of_clustering/figures/')

# %%
# -----------------------------------------------------------------------
# Load the data
# -----------------------------------------------------------------------

categories_test = [
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

fetched_data = fetch_20newsgroups(
    subset='all',
    categories=categories_paper,
    shuffle=True,
    remove=('headers', 'footers', 'quotes')
)

y = fetched_data.target
ground_k = len(np.unique(y))
print(f"The ground truth number of clusters is: {ground_k}")

# Map target labels to category names for visualization
category_labels = [fetched_data.target_names[target] for target in y]
hover_df = pd.DataFrame(category_labels, columns=['category'])

print("Category labels and indices:")
for idx, cat in enumerate(fetched_data.target_names):
    print(f"{idx}: {cat}")

# %%
# -----------------------------------------------------------------------
# Text pre-processing: Lemmatization
# -----------------------------------------------------------------------

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_text(doc):
    tokens = word_tokenize(doc)
    return ' '.join(lemmatizer.lemmatize(token) for token in tokens)

# Use list comprehension instead of in-place modification for safety and clarity
lemmatized_docs = [lemmatize_text(doc) for doc in fetched_data.data]

# %%
# -----------------------------------------------------------------------
# TF-IDF vectorization
# -----------------------------------------------------------------------

vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words='english', min_df=5)
tfidf_X = vectorizer.fit_transform(lemmatized_docs)

print(f"Shape of tf-idf matrix: {tfidf_X.shape}")

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# %%
# -----------------------------------------------------------------------
# Get top keywords for each document
# -----------------------------------------------------------------------
# Function to get top N keywords for a single sparse vector row
def get_top_n_keywords(row, feature_names, n=10):
    # row is sparse vector, get dense representation
    row_array = row.toarray().flatten()
    # Get indices of top n tfidf scores
    top_n_idx = row_array.argsort()[-n:][::-1]
    # Get the corresponding feature names
    top_keywords = [feature_names[i] for i in top_n_idx if row_array[i] > 0]
    return ', '.join(top_keywords)

# Apply to all documents (tfidf_X is sparse matrix)
top_keywords_list = [get_top_n_keywords(tfidf_X[i], feature_names, n=10) for i in range(tfidf_X.shape[0])]

# %%
# -----------------------------------------------------------------------
# Dimension reduction with UMAP and visualization
# -----------------------------------------------------------------------

# UMAP embedding to 10 dimensions (for later downstream tasks)
reducer = umap.UMAP(n_components=10, metric='cosine') # , random_state=42
embedding = reducer.fit_transform(tfidf_X)

# Plot first two components coloured by cluster labels
plt.figure(figsize=(12, 6))
sns.scatterplot(
    x=embedding[:, 0],
    y=embedding[:, 1],
    hue=y,
    palette=color_map,
    legend='full'
)
plt.title('UMAP projection to 10D - visualized in 2D (first two components)')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.gca().set_aspect('equal', 'datalim')
plt.show()

# %%
# UMAP embedding for visualization only (2D)
embedding_vis = umap.UMAP(n_components=2, metric='cosine', random_state=42).fit(tfidf_X)
umap.plot.points(embedding_vis, labels=hover_df['category'])
plt.gca().set_aspect('equal', 'datalim')
plt.show()

# Extract the 2D embedding coordinates
embedding_coords = embedding_vis.embedding_

# Create a DataFrame with coordinates and labels
embedding_vis_df = pd.DataFrame({
    'UMAP_1': embedding_coords[:, 0],
    'UMAP_2': embedding_coords[:, 1],
    'category': hover_df['category']
})

# add the keywords to the DataFrame
embedding_vis_df['top_keywords'] = top_keywords_list

# Add the original email body text (full content) to the DataFrame for hover display
embedding_vis_df['email_body'] = fetched_data.data

# %% Save to CSV (or choose any preferred path)
output_file_embeddings = os.path.join(SAVE_DATA_DIR, "6NewsgroupsUMAP2_embeddings.csv")
embedding_vis_df.to_csv(output_file_embeddings, index=False)

# %% save the UMAP embedding for clustering
embedding_df = pd.DataFrame(embedding, columns=[f'UMAP_{i+1}' for i in range(embedding.shape[1])])

# Add true labels to the DataFrame
embedding_df['class'] = y

# %% write the DataFrame to a CSV file for later use
# Define output file path
output_file = os.path.join(SAVE_DATA_DIR, "6NewsgroupsUMAP10_with_class.csv")

# Save DataFrame to CSV (without the DataFrame index)
embedding_df.to_csv(output_file, index=False)

print(f"Saved embedding and labels DataFrame to: {output_file}")
