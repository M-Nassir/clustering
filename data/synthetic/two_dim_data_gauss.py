# %% imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# %%
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
#
#                         Make the two dimensional synthetic data
#
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

def generate_clustering_2d_gauss_data(
        n_samples=10000,
        n_components=5,
        num_features=2,
        rand_seed=0,
        same_density=False,
        labelled_fraction=0.01,
        add_anomaly_cluster=True,
        plot=True,
    ):

    np.random.seed(rand_seed)

    # Define cluster standard deviations
    if same_density:
        std_dev = 0.6
    else:
        # Set different std deviations for each component
        std_dev = [2.5, 0.8, 1.2, 5, 0.4][:n_components]

    # Generate the main clusters
    X, y_true = make_blobs(
        n_samples=n_samples,
        centers=n_components,
        n_features=num_features,
        cluster_std=std_dev,
        random_state=rand_seed
    )

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(num_features)])
    df['y_true'] = y_true
    df['y_live'] = -1  # Default to unlabelled

    # Randomly label a small fraction of points
    labelled_indices = np.random.choice(df.index, size=int(n_samples *              
                                                           labelled_fraction), replace=False)
    df.loc[labelled_indices, 'y_live'] = df.loc[labelled_indices, 'y_true']

    # Print stats
    n_labelled = len(labelled_indices)
    p_labelled = 100 * n_labelled / len(df)
    print(f"Number of labelled examples: {n_labelled}")
    print(f"Number of unlabelled examples: {len(df) - n_labelled}")
    print(f"Percentage of labelled data: {p_labelled:.2f}%")

    # Add anomaly cluster if requested
    if add_anomaly_cluster:
        X_anom, _ = make_blobs(
            n_samples=300,
            centers=[list(range(num_features)), list(range(num_features))],
            n_features=num_features,
            cluster_std=[8.6, 0.2],
            random_state=rand_seed + 1
        )
        df_anom = pd.DataFrame(X_anom, columns=[f"f{i}" for i in range(num_features)])
        df_anom['y_true'] = -1
        df_anom['y_live'] = -1
        df = pd.concat([df, df_anom], ignore_index=True)

    return df

# %% plot the data

def plot_clustering_results_2d(df, title_prefix="Synthetic 2D Clustering"):
    """
    Plots clustering results:
    1. All points (including anomalies)
    2. All points excluding anomalies
    3. Only labelled samples

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'f0', 'f1', 'y_true', and 'y_live'.
        title_prefix (str): Prefix for plot titles.
    """
    # CUD colormap has distinct and accessible colors
    cud_palette = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

    # Add color for noise (-1)
    colors = {-1: 'red'}  # Noise is marked with red
    # Generate a dictionary of color mappings for clusters (0 to 39)
    colors.update({i: cud_palette[i % len(cud_palette)] for i in range(40)})

    # 1. Plot all points including anomalies
    plt.figure(figsize=(12, 6))
    ax = sns.scatterplot(
        data=df, x='f0', y='f1', hue='y_true',
        palette=colors, s=20, edgecolor='none'
    )
    ax.legend(title='True Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"{title_prefix} (All Points)")
    plt.tight_layout()
    plt.show()

    # 2. Plot all points excluding anomalies (y_true != -1)
    non_anomalous = df[df['y_true'] != -1]
    if not non_anomalous.empty:
        plt.figure(figsize=(12, 6))
        ax = sns.scatterplot(
            data=non_anomalous, x='f0', y='f1', hue='y_true',
            palette=colors, s=20, edgecolor='none'
        )
        ax.legend(title='True Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"{title_prefix} (Without Anomalies)")
        plt.tight_layout()
        plt.show()
    else:
        print("No non-anomalous samples to plot.")

    # 3. Plot only labelled samples (y_live != -1)
    labelled_data = df[df['y_live'] != -1]
    if not labelled_data.empty:
        plt.figure(figsize=(12, 6))
        ax = sns.scatterplot(
            data=labelled_data, x='f0', y='f1', hue='y_true',
            palette=colors, s=20, edgecolor='none'
        )
        ax.legend(title='Labelled Samples', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"{title_prefix} (Labelled Samples Only)")
        plt.tight_layout()
        plt.show()
    else:
        print("No labelled samples to plot.")

# %% Example usage:
# data_2d = generate_clustering_2d_gauss_data(rand_seed=1)
# plot_clustering_results(data_2d)

