import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap  # Import UMAP for dimensionality reduction

def plot_clusters(df, feature_columns, label_column, title=None, colors=None, show_seeds_only=False):
    """
    Generic cluster plotting function for 1D, 2D, or multi-dimensional data.

    Parameters:
    - df (pd.DataFrame): The DataFrame with features and label column.
    - feature_columns (list): List of feature column names.
    - label_column (str): Column name containing cluster labels to plot.
    - title (str): Plot title (optional).
    - colors (dict): Optional custom colour palette (label: color).
    - show_seeds_only (bool): Whether to show only seed points (exclude noise).
    """

    # Set colors
    if colors is None:
        cud_palette = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', 
                       '#0072B2', '#D55E00', '#CC79A7']
        unique_labels = sorted(df[label_column].unique())
        n_labels = len([label for label in unique_labels if label != -1])
        colors = {-1: 'red'}
        if n_labels <= len(cud_palette):
            cluster_palette = cud_palette[:n_labels]
        else:
            cluster_palette = sns.color_palette("hls", n_colors=n_labels)
        valid_labels = [label for label in unique_labels if label != -1]
        colors.update({label: cluster_palette[i] for i, label in enumerate(valid_labels)})

    # Handle multi-dimensional data
    if len(feature_columns) > 2:
        # Apply UMAP dimensionality reduction
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(df[feature_columns])
        df['UMAP_1'] = reduced_features[:, 0]
        df['UMAP_2'] = reduced_features[:, 1]
        feature_columns = ['UMAP_1', 'UMAP_2']

    if len(feature_columns) == 1:
        # 1D plotting
        xcol = feature_columns[0]
        fig, ax = plt.subplots(figsize=(12, 6))
        if not show_seeds_only:
            sns.histplot(df, x=xcol, bins=1000, color='lightgrey', ax=ax)
            ax.set_ylabel('Frequency')
            ax.set_xlabel('')
        plot_df = df.copy()
        if show_seeds_only:
            plot_df = plot_df[plot_df[label_column] != -1]
        sns.scatterplot(
            x=plot_df[xcol],
            y=np.full_like(plot_df[xcol], df[xcol].max() * 0.01),
            hue=plot_df[label_column],
            palette=colors,
            ax=ax,
            legend='full',
            edgecolor='none'
        )
    elif len(feature_columns) == 2:
        # 2D plotting
        plot_df = df.copy()
        if show_seeds_only:
            plot_df = plot_df[plot_df[label_column] != -1]
        fig, ax = plt.subplots(figsize=(18, 10))
        sns.scatterplot(
            x=plot_df[feature_columns[0]],
            y=plot_df[feature_columns[1]],
            hue=plot_df[label_column],
            palette=colors,
            edgecolor='none'
        )
        plt.xlabel(feature_columns[0])
        plt.ylabel(feature_columns[1])
    else:
        raise ValueError("Error: feature_columns must contain at least 1 column")

    plt.title(title or f"Clustering by {label_column}")
    plt.legend(title=label_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


