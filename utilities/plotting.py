import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_clusters(df, feature_columns, label_column, title=None, colors=None, show_seeds_only=False):
    """
    Generic cluster plotting function for 1D or 2D data.

    Parameters:
    - df (pd.DataFrame): The DataFrame with features and label column.
    - feature_columns (list): List of one or two feature column names.
    - label_column (str): Column name containing cluster labels to plot.
    - title (str): Plot title (optional).
    - colors (dict): Optional custom colour palette (label: color).
    """

    # set colors
    if colors is None:
        # CUD colormap: distinct and accessible colours
        cud_palette = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', 
                       '#0072B2', '#D55E00', '#CC79A7']

        # Determine unique labels (including noise)
        unique_labels = sorted(df[label_column].unique())
        n_labels = len([label for label in unique_labels if label != -1])

        # Initialise with red for noise
        colors = {-1: 'red'}

        # Use cud_palette if label count is small, else generate full palette
        if n_labels <= len(cud_palette):
            cluster_palette = cud_palette[:n_labels]
        else:
            cluster_palette = sns.color_palette("hls", n_colors=n_labels)

        # Assign each label (except -1) a colour
        valid_labels = [label for label in unique_labels if label != -1]
        colors.update({label: cluster_palette[i] for i, label in enumerate(valid_labels)})

    if len(feature_columns) == 1:
        # 1D plotting with histogram and overlay
        xcol = feature_columns[0]
        fig, ax = plt.subplots(figsize=(12, 6))

        if not show_seeds_only:
            # Plot histogram
            sns.histplot(df, x=xcol, bins=1000, color='lightgrey', ax=ax)
            ax.set_ylabel('Frequency')
            ax.set_xlabel('')
        
        # Prepare data to plot (exclude anomalies if requested)
        plot_df = df.copy()
        if show_seeds_only:
            plot_df = plot_df[plot_df[label_column] != -1]

        # Overlay scatterplot of cluster labels
        sns.scatterplot(
            x=plot_df[xcol],
            y=np.full_like(plot_df[xcol], df[xcol].max() * 0.01),
            hue=plot_df[label_column],
            palette=colors,
            ax=ax,
            legend='full',
            edgecolor='none'
        );
        
    elif len(feature_columns) >= 2:
        # 2D plotting
        sns.scatterplot(
            x=df[feature_columns[0]],
            y=df[feature_columns[1]],
            hue=df[label_column],
            palette=colors,
            edgecolor='none'
        )
        plt.xlabel(feature_columns[0])
        plt.ylabel(feature_columns[1])
    else:
        raise ValueError("feature_columns must contain at least 1 column")

    plt.title(title or f"Clustering by {label_column}")
    plt.legend(title=label_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


