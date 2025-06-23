# %%
# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import umap
import matplotlib.pyplot as plt

# Paths
CURRENT_DIR = os.getcwd()
ROOT_PATH = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
SAVE_DATA_DIR = os.path.join(ROOT_PATH, "data", 'processed')

# Create directory if needed
os.makedirs(SAVE_DATA_DIR, exist_ok=True)

# %%
# -----------------------------------------------------------------------
# Load the MNIST digits dataset
# -----------------------------------------------------------------------

digits = load_digits()
X = digits.data          # shape (1797, 64) - 8x8 pixel images flattened
y = digits.target        # labels 0-9

print(f"Loaded MNIST digits dataset with shape {X.shape} and labels shape {y.shape}")

# %%
# -----------------------------------------------------------------------
# UMAP embedding: 10 dimensions for downstream tasks
# -----------------------------------------------------------------------

reducer_10d = umap.UMAP(n_components=10, metric='euclidean', random_state=42)
embedding_10d = reducer_10d.fit_transform(X)

# %%
# -----------------------------------------------------------------------
# UMAP embedding: 2D for visualization
# -----------------------------------------------------------------------

reducer_2d = umap.UMAP(n_components=2, metric='euclidean', random_state=42)
embedding_2d = reducer_2d.fit_transform(X)

# %%
# -----------------------------------------------------------------------
# Prepare DataFrame for embeddings and images
# -----------------------------------------------------------------------

# Convert images to a format that can be saved easily, e.g. list or bytes
# Here we'll store as lists of pixel values for easy re-loading later
images_list = [img.reshape(8, 8) for img in X]

# Create DataFrame with 2D visualization embeddings and labels
df_vis = pd.DataFrame({
    'UMAP_1': embedding_2d[:, 0],
    'UMAP_2': embedding_2d[:, 1],
    'class': y,
    # store the flattened image pixels as lists for convenience
    'image_pixels': [img.tolist() for img in X]
})

# Also save the 10D embeddings and labels for downstream use
df_10d = pd.DataFrame(embedding_10d, columns=[f'UMAP_{i+1}' for i in range(embedding_10d.shape[1])])
df_10d['class'] = y

# %%
# -----------------------------------------------------------------------
# Save DataFrames for later use
# -----------------------------------------------------------------------

output_vis_file = os.path.join(SAVE_DATA_DIR, "MNIST_UMAP2_with_images.csv")
df_vis.to_csv(output_vis_file, index=False)

output_10d_file = os.path.join(SAVE_DATA_DIR, "MNIST_UMAP10_with_class.csv")
df_10d.to_csv(output_10d_file, index=False)

print(f"Saved 2D visualization DataFrame with images to: {output_vis_file}")
print(f"Saved 10D embedding DataFrame with labels to: {output_10d_file}")

# %%
# -----------------------------------------------------------------------
# Optional: plot the 2D embedding colored by digit label
# -----------------------------------------------------------------------

import seaborn as sns

plt.figure(figsize=(10, 8))
sns.scatterplot(x='UMAP_1', y='UMAP_2', hue='class', palette='tab10', data=df_vis, legend='full')
plt.title('UMAP 2D projection of MNIST digits')
plt.gca().set_aspect('equal', 'datalim')
plt.show()

