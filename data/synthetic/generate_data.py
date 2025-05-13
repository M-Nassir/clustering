# %% imports
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

# %% -----------------------------------------------------------------------
#                         Generate 1D Clustered Data (Manual Design)
# --------------------------------------------------------------------------

def generate_clustering_1d_data(repeat_const=100, percent_labelled=0.03, random_state=None):
    """
    Generates a hand-crafted 1D dataset with repeated clusters, anomalies, and partial labels.

    Parameters:
    - repeat_const (int): Number of times to repeat base data for density.
    - percent_labelled (float): Fraction of data to mark as labelled.
    - random_state (int): Seed for reproducibility.

    Returns:
    - df (pd.DataFrame): DataFrame containing 'X', 'y_true', and 'y_live'.
    """

    # Define base cluster data (labelled)
    data_main = np.array([
        [2.1, 0],
        [2.6, 0],
        [2.4, 0],
        [2.5, 0],
        [2.3, 0],
        [2.1, 0],
        [2.3, 0],
        [2.6, 0],
        [2.6, 0],
        [2.0, 0],
        [2.1, 0],
        [2.0, 0],
        [1.9, 0],
        [2.1, 0],
        [1.8, 0],
        [2.9, 0],

        [56, 1],
        [55, 1],
        [56, 1],
        [58, 1],
        [59, 1],
        [57, 1],
        [56, 1],
        [55, 1],
        [55, 1],
        [55, 1],
        [56.3, 1],
        [55.3, 1],
        [51, 1],
        [56, 1],
        [54.4, 1],
        [57, 1],
        [56, 1],
        [52, 1],
        [53, 1],
        [51, 1],
        [51, 1],
        [50, 1],

        [100, 2],
        [101, 2],
        [102, 2],
        [105, 2],
        [110, 2],
        [108, 2],
        [107, 2],
        [106, 2],
        [111, 2],

        [100, 2],
        [103, 2],
        [101.3, 2],
        [101.8, 2],
        [101.2, 2],
        [109, 2],
        [108, 2],
        [108, 2],
        [111, 2],
        [111, 2],
    ])

    # Anomalies and mislabelled points
    data_anomalies_mislablled = np.array([
        [61, 1],
        [58, 1],
        [8.2, -1],
        [8.3, -1],
        [25, -1],
        [40, -1],
        [80, -1],
        [95, -1],
        [112, 2],
    ])

    # Repeat main data
    data_main_repeated = np.repeat(data_main, repeat_const, axis=0)

    # Combine with anomalies
    data = np.concatenate((data_main_repeated, data_anomalies_mislablled), axis=0)

    # Shuffle rows
    np.random.shuffle(data)

    # Create DataFrame
    df = pd.DataFrame(data, columns=['X', 'y_true'])
    df['y_true'] = df['y_true'].astype(int)

    # Add small noise
    noise = np.round(np.random.normal(0, 1, df.shape[0]), 1) * 0.1
    df['X'] += noise

    # Assign partial labels to simulate semi-supervised setup using: percent_labelled
    mask = np.random.choice(np.arange(len(df)), size=int(len(df) * percent_labelled), replace=False)
    df['y_live'] = -1
    df.loc[mask, 'y_live'] = df.loc[mask, 'y_true']

    # create an unknown anomaly cluster after the random labelling
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

    df_extra = pd.DataFrame(data_anomaly_cluster, columns=['X', 'y_true', 'y_live'])
    df = pd.concat([df, df_extra], ignore_index=True)

    return df

# %% -----------------------------------------------------------------------
#                         Generate 1D Gaussian Clusters with Anomalies
# --------------------------------------------------------------------------

def generate_clustering_1d_gauss_anomalies(random_seed=None,
                                            labelled_percent=0.5,
                                            cluster_params=None,
                                            samples_per_cluster=10000,
                                            include_anomaly_cluster=True):
    """
    Generate 1D synthetic data using Gaussian blobs and inject anomalies.

    Parameters:
    - random_seed (int): For reproducibility.
    - labelled_percent (float): Percentage of points to label.
    - cluster_params (list): (mean, std) tuples for clusters.
    - samples_per_cluster (int): Samples per Gaussian cluster.
    - include_anomaly_cluster (bool): Whether to include a dense anomaly cluster.

    Returns:
    - df (DataFrame): Data with 'X', 'y_true', 'y_live' columns.
    """

    # Generate Gaussian clusters
    data_holder = []
    for i, (mu, sig) in enumerate(cluster_params):
        X = np.random.normal(loc=mu, scale=sig, size=samples_per_cluster).reshape(-1, 1)
        y = np.full((samples_per_cluster, 1), i)
        data_holder.append(np.hstack([X, y]))

    data_main = np.vstack(data_holder)

    # Inject predefined anomalies/mislabelled points
    anomalies_manual = np.array([
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
    data = np.vstack([data_main, anomalies_manual])

    # Shuffle and convert to DataFrame
    np.random.shuffle(data)
    df = pd.DataFrame(data, columns=['X', 'y_true'])
    df['y_true'] = df['y_true'].astype(int)

    # Add noise
    noise = np.round(np.random.normal(0, 1, df.shape[0]), 1) * 0.1
    df['X'] += noise

    # Label a small percentage of data
    p = labelled_percent / 100
    num_labelled = int(p * len(df))
    labelled_indices = np.random.choice(df.index, size=num_labelled, replace=False)

    df['y_live'] = -1
    df.loc[labelled_indices, 'y_live'] = df.loc[labelled_indices, 'y_true']

    # Optionally, add an unknown anomaly cluster far from existing clusters
    if include_anomaly_cluster:
        anomaly_cluster = np.array([
            [150, -1, -1], [151, -1, -1], [152, -1, -1], [155, -1, -1],
            [156, -1, -1], [157, -1, -1], [158, -1, -1], [159, -1, -1],
            [151, -1, -1], [150, -1, -1], [151, -1, -1], [152, -1, -1],
            [155, -1, -1], [156, -1, -1], [157, -1, -1], [158, -1, -1],
            [159, -1, -1], [151, -1, -1],
        ])
        df_anomaly = pd.DataFrame(anomaly_cluster, columns=['X', 'y_true', 'y_live'])
        df = pd.concat([df, df_anomaly], ignore_index=True)

    # Summary
    num_unlabelled = (df['y_live'] == -1).sum()
    num_labelled = (df['y_live'] != -1).sum()
    labelled_pct = round(num_labelled / df.shape[0] * 100, 2)

    print(f"Number of unlabelled examples: {num_unlabelled}")
    print(f"Number of labelled examples: {num_labelled}")
    print(f"Percentage of labelled data: {labelled_pct}%")

    return df

# %% -----------------------------------------------------------------------
#                         Generate 2D Gaussian Data
# --------------------------------------------------------------------------

def generate_clustering_2d_gauss_data(
        n_samples=10000,
        n_components=5,
        num_features=2,
        rand_seed=None,
        same_density=False,
        labelled_fraction=0.01,
        add_anomaly_cluster=True,
        std_dev=None,
    ):
    """
    Generate 2D synthetic dataset using Gaussian blobs with optional anomaly injection.

    Parameters:
    - n_samples (int): Number of main data samples.
    - n_components (int): Number of Gaussian blobs.
    - num_features (int): Dimensionality (default 2D).
    - rand_seed (int): Random seed.
    - same_density (bool): Use identical standard deviation for all clusters.
    - labelled_fraction (float): Fraction of samples to label.
    - add_anomaly_cluster (bool): Whether to inject an anomaly cluster.

    Returns:
    - df (pd.DataFrame): DataFrame with 'f0', ..., 'fN', 'y_true', and 'y_live'.
    """

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