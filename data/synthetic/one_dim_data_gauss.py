# %% imports
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# %%
def generate_clustering_1d_gauss_anomalies(random_seed=42,
                                               labelled_percent=0.5,
                                               cluster_params=[(0, 1), (50, 3), (100, 6)],
                                               samples_per_cluster=10000,
                                               include_anomaly_cluster=True):
    """
    Generates a synthetic 1D clustering dataset with Gaussian clusters,
    injected anomalies, and semi-supervised labels.

    Parameters:
    - seed (int): Random seed for reproducibility.
    - labelled_percent (float): Percent of data to label (e.g., 0.5 for 0.5%).
    - cluster_params (list): List of (mean, std) tuples for clusters.
    - samples_per_cluster (int): Number of samples per cluster.
    - include_anomaly_cluster (bool): If True, adds a dense outlier cluster far from the rest.

    Returns:
    - df (DataFrame): Contains 'X', 'y_true', 'y_live' columns.
    """

    np.random.seed(random_seed)

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
