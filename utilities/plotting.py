import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import umap  
import os

def plot_clusters(df, feature_columns, label_column, 
                  title=None, 
                  x_axis_label='', y_axis_label=None, legend_label='Cluster labels',
                  colors=None, show_seeds_only=False):
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

    # set figure sizes
    fig_height = 8
    fig_width = 12
    
    # Set colors
    if colors is None:
        cud_palette = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', 
                    '#0072B2', '#D55E00', '#CC79A7']
        unique_labels = sorted(df[label_column].unique())
        n_labels = len([label for label in unique_labels if label != -1])

        # Base colors dict for labels -1 and 0..10
        colors = {
            -1: 'red', 0: 'green', 1: 'blue', 2: 'black',
            3: 'orange', 4: 'purple', 5: 'brown',
            6: 'pink', 7: 'cyan', 8: 'darkblue',
            9: 'violet', 10: 'magenta',
        }

        # Determine palette to use for labels >= 11 or if n_labels > len(colors)
        if n_labels <= len(colors):
            cluster_palette = cud_palette[:n_labels]
        else:
            cluster_palette = sns.color_palette("hls", n_colors=n_labels)

        # Get labels excluding -1 (anomaly or unassigned)
        valid_labels = [label for label in unique_labels if label != -1]

        # Add or update colors for labels not already in colors dict
        for i, label in enumerate(valid_labels):
            if label not in colors:
                # Use color from cluster_palette; cycle if needed
                colors[label] = cluster_palette[i % len(cluster_palette)]

    plot_df = df.copy()
    # Handle multi-dimensional data
    if len(feature_columns) > 2:
        # Apply UMAP dimensionality reduction
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(df[feature_columns])
        plot_df['UMAP_1'] = reduced_features[:, 0]
        plot_df['UMAP_2'] = reduced_features[:, 1]
        feature_columns = ['UMAP_1', 'UMAP_2']

    if len(feature_columns) == 1:
        # 1D plotting
        xcol = feature_columns[0]

        # set the figure
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))  # figsize expects (width, height)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # If only showing seeds, filter out unlabelled points
        if show_seeds_only:
            plot_df = plot_df[plot_df[label_column] != -1]

        if show_seeds_only is False:
            sns.histplot(plot_df, x=xcol, bins=1000, color='lightgrey', ax=ax)

        # Distinct marker shapes for clusters:
        # marker_list = ['o', 'X', 'p', '^', 'v', '<', '>', 'P', '*', 'X', 'H', 'd', '8', 'p', 'h']
        # markers = {label: marker_list[i % len(marker_list)] for i, label in enumerate(unique_labels)}

        sns.scatterplot(
            data=plot_df,  # <- required when using column name strings
            x=xcol,
            y=np.full_like(plot_df[xcol], plot_df[xcol].max() * 0.03),
            hue=label_column,
            # style=label_column,
            # markers=markers,
            palette=colors,
            ax=ax,
            legend='full',
            edgecolor='none'           
        )

        # Explicitly override axis labels *after* plotting
        ax.set_xlabel('' if x_axis_label is None else x_axis_label)
        if y_axis_label is not None:
            ax.set_ylabel(y_axis_label)

    elif len(feature_columns) == 2:
        # 2D plotting       
        if show_seeds_only:
            plot_df = plot_df[plot_df[label_column] != -1]

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        sns.scatterplot(
            x=plot_df[feature_columns[0]],
            y=plot_df[feature_columns[1]],
            hue=plot_df[label_column],
            palette=colors,
            edgecolor='white',
        )
        plt.xlabel('')
        plt.ylabel('')
    else:
        raise ValueError("Error: feature_columns must contain at least 1 column")

    plt.xlabel(plt.gca().get_xlabel(), fontsize=16)
    plt.ylabel(plt.gca().get_ylabel(), fontsize=16)

    plt.title('')
    plt.legend(title=legend_label, title_fontsize=16, fontsize=14, 
               bbox_to_anchor=(1.05, 1), loc='upper right',
               markerscale=2.0,)
    plt.tight_layout()
    plt.show()

    return fig

def plot_enabled_clusterings(df, clustering_flags, feature_columns, 
                              plot_save_path=None, dataset_name=None, save_plots=False):
    for name in clustering_flags:
        if clustering_flags.get(name, False):
            fig = plot_clusters(df, feature_columns, 
                                label_column=name, title=f"{name} Clustering")

            if save_plots is True and plot_save_path is not None:
                filename = f"{name}_{dataset_name}.png"
                filepath = os.path.join(plot_save_path, filename)
                fig.savefig(filepath)
                print(f"Saved plot to {filepath}")

def plot_confusion_matrices_for_clustering(df, true_label_col, clustering_flags, 
                                           title_prefix="Confusion Matrix", exclude_noise=False):
    """
    Plots confusion matrices for all enabled clustering methods in `clustering_flags`.

    Parameters:
    - df (pd.DataFrame): DataFrame with true and predicted labels.
    - true_label_col (str): Name of the column containing ground-truth labels.
    - clustering_flags (dict): Dictionary of clustering method names with booleans indicating if they are enabled.
    - title_prefix (str): Prefix for plot titles.
    - exclude_noise (bool): Whether to exclude the noise label (-1) from the confusion matrix.
    """
    true_labels = df[true_label_col].values

    clustering_methods = [name for name, enabled in clustering_flags.items() if enabled]

    for method in clustering_methods:
        predicted_labels = df[method].values

        print(f"Unique predicted labels for {method}: {np.unique(predicted_labels)}")

        # Build label set
        labels = sorted(np.unique(np.concatenate((true_labels, predicted_labels))))
        if exclude_noise and -1 in labels:
            labels.remove(-1)

        # Compute confusion matrix with consistent label ordering
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

        # Print as DataFrame for reference
        print(f"\nConfusion matrix for {method}:")
        print(pd.DataFrame(cm, index=labels, columns=labels))

        # Plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title(f"{title_prefix} - {method}")
        plt.tight_layout()
        plt.show()