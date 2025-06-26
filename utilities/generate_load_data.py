# %% imports
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import os
from sklearn.preprocessing import StandardScaler
import logging

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
                                            labelled_percent=0.1,
                                            cluster_params=[(0, 1), (50, 3), (100, 8)],   
                                            samples_per_cluster=[10000, 10000, 10000],
                                            include_anomaly_cluster=True):
    """
    Generate 1D synthetic data using Gaussian blobs and inject anomalies.

    Parameters:
    - random_seed (int): For reproducibility.
    - labelled_percent (float): Percentage of points to label.
    - cluster_params (list): (mean, std) tuples for clusters.
    - samples_per_cluster (list): Samples per Gaussian cluster.
    - include_anomaly_cluster (bool): Whether to include a dense anomaly cluster.

    Returns:
    - df (DataFrame): Data with 'X', 'y_true', 'y_live' columns.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate Gaussian clusters
    data_holder = []
    for i, (mu, sig) in enumerate(cluster_params):
        X = np.random.normal(loc=mu, scale=sig, size=samples_per_cluster[i]).reshape(-1, 1)
        y = np.full((samples_per_cluster[i], 1), i)
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

    # Label a small, class-balanced percentage of data
    df['y_live'] = -1
    true_classes = df[df['y_true'] >= 0]['y_true'].unique()
    labelled_total = int((labelled_percent / 100) * len(df))

    # Ensure at least one per class, and balance the rest
    labelled_per_class = max(1, labelled_total // len(true_classes))

    labelled_indices = []
    for cls in true_classes:
        cls_indices = df[(df['y_true'] == cls)].index
        n_samples = min(labelled_per_class, len(cls_indices))
        labelled_indices.extend(np.random.choice(cls_indices, size=n_samples, replace=False))

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

    logging.info("Number of unlabelled examples: %s", num_unlabelled)
    logging.info("Number of labelled examples: %s", num_labelled)
    logging.info("Percentage of labelled data: %s%%", labelled_pct)


    return df


# %% -----------------------------------------------------------------------
#                         Generate 2D Gaussian Data
# --------------------------------------------------------------------------

# # Experimental function WIP
# def generate_clustering_2d_gauss_data_wip( 
#         # when taking samples of seed points need to take enough of the smaller clusters, as taking random
#         # sample means more of the larger clusters are sampled. But the novel clustering still works very well
#         # here it seems; also need to correctly handle the anomaly class for k-means, etc. need to look through
#         # this code in detail
                                          
#         n_samples=10000,
#         n_components=8,
#         num_features=2,
#         rand_seed=None,
#         same_density=False,
#         labelled_fraction=0.01,
#         add_anomaly_cluster=True,
#         std_dev=None,
#     ):
#     """
#     Generate 2D synthetic dataset using Gaussian blobs with optional anomaly injection.

#     Returns:
#     - df (pd.DataFrame): DataFrame with 'f0', ..., 'fN', 'y_true', and 'y_live'.
#     """

#     np.random.seed(rand_seed)

#     # Strongly imbalanced cluster sizes
#     cluster_sizes = [150, 200, 500, 800, 1350, 1800, 2500, 2700]
#     assert sum(cluster_sizes) == n_samples, "Cluster sizes must sum to n_samples"

#     X_list, y_list = [], []

#     # Slightly wider spread of cluster centers
#     center_box = (-20, 20)
#     centers = [np.random.uniform(center_box[0], center_box[1], size=num_features)
#                for _ in range(n_components)]

#     for i, (size, center) in enumerate(zip(cluster_sizes, centers)):
#         # Vary spread to break density assumptions
#         std = std_dev if same_density else np.random.uniform(0.8, 3.0)
#         X_blob, _ = make_blobs(
#             n_samples=size,
#             centers=[center],
#             n_features=num_features,
#             cluster_std=std,
#             random_state=(None if rand_seed is None else rand_seed + i)
#         )
#         X_list.append(X_blob)
#         y_list.append(np.full(size, i))

#     X = np.vstack(X_list)
#     y_true = np.concatenate(y_list)

#     df = pd.DataFrame(X, columns=[f"f{i}" for i in range(num_features)])
#     df['y_true'] = y_true
#     df['y_live'] = -1

#     # Label a small random fraction of points
#     labelled_indices = np.random.choice(df.index, size=int(n_samples * labelled_fraction), replace=False)
#     df.loc[labelled_indices, 'y_live'] = df.loc[labelled_indices, 'y_true']

#     # Label stats
#     n_labelled = len(labelled_indices)
#     p_labelled = 100 * n_labelled / len(df)
#     logging.info("Number of labelled examples: %s", n_labelled)
#     logging.info("Number of unlabelled examples: %s", len(df) - n_labelled)
#     logging.info("Percentage of labelled data: %.2f%%", p_labelled)

#     # Add anomalies nearby but not inside existing clusters
#     if add_anomaly_cluster:
#         anomaly_centers = [
#             np.array([15, 15]),
#             np.array([-12, -14]),
#             np.array([8, -18])
#         ]
#         anomaly_stds = [1.0, 1.5, 2.0]

#         X_anom = []
#         for c, s in zip(anomaly_centers, anomaly_stds):
#             blob, _ = make_blobs(n_samples=100, centers=[c], n_features=num_features,
#                                  cluster_std=s, random_state=rand_seed)
#             X_anom.append(blob)

#         X_anom = np.vstack(X_anom)
#         df_anom = pd.DataFrame(X_anom, columns=[f"f{i}" for i in range(num_features)])
#         df_anom['y_true'] = -1
#         df_anom['y_live'] = -1

#         df = pd.concat([df, df_anom], ignore_index=True)

#     return df

def generate_clustering_2d_gauss_data(
        n_samples=10000,
        n_components=8,
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

    np.random.seed(rand_seed)
    
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
    logging.info("Number of labelled examples: %s", n_labelled)
    logging.info("Number of unlabelled examples: %s", len(df) - n_labelled)
    logging.info("Percentage of labelled data: %.2f%%", p_labelled)

    # Add anomaly cluster if requested
    if add_anomaly_cluster:
        X_anom, _ = make_blobs(
            n_samples=300,
            centers=[(10, 10), (10, 20), (0, 10)],#[list(range(num_features)), list(range(num_features))],
            n_features=num_features,
            cluster_std=[8.6, 0.2, 10],
            random_state=rand_seed+1
        )
        df_anom = pd.DataFrame(X_anom, columns=[f"f{i}" for i in range(num_features)])
        df_anom['y_true'] = -1
        df_anom['y_live'] = -1
        df = pd.concat([df, df_anom], ignore_index=True)

    return df


# n_samples=300, centers=[(10, 10), (10, 20), (0, 10)], n_features=num_features,
#     cluster_std=[8.6, 0.2, 10], random_state=0,

def prepare_and_seed_dataset(dataset_name, percent_labelled, k, random_seed, label_column):
    # Read data from the processed folder CSV
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    csv_file_path = os.path.join(project_root, "data", "processed", f"{dataset_name}.csv")
    df = pd.read_csv(csv_file_path)

    # Ensure label column exists
    if label_column not in df.columns:
        raise ValueError(f"The specified label column '{label_column}' was not found in the dataset.")

    # Rename label column to 'y_true'
    df.rename(columns={label_column: 'y_true'}, inplace=True)

    # Determine number of unique classes
    if k is None:
        k = df['y_true'].nunique()

    # Number of labelled samples to select
    n_labelled = int(len(df) * percent_labelled)
    n_labelled = min(n_labelled, len(df))

    # Initialize y_live column as unlabelled (-1)
    df['y_live'] = -1

    # Use local random generator for reproducibility
    rng = np.random.default_rng(random_seed)
    logging.debug("Random seed used for sampling: %s", random_seed)


    # Balanced sampling across classes
    classes = df['y_true'].unique()
    n_classes = len(classes)
    n_per_class = max(n_labelled // n_classes, 1)

    labelled_indices = []

    for cls in classes:
        class_indices = df[df['y_true'] == cls].index
        n_samples = min(n_per_class, len(class_indices))
        sampled = rng.choice(class_indices, size=n_samples, replace=False)
        labelled_indices.extend(sampled)

    # Assign true labels to y_live for sampled indices
    df.loc[labelled_indices, 'y_live'] = df.loc[labelled_indices, 'y_true']

    return df, k


def load_dataset(dataset_name, random_seed, k, percent_labelled, standardise):
    """
    Loads a specified dataset, optionally standardizing its features.

    Args:
        dataset_name (str): The name of the dataset to load (e.g., "1d_simple", "2d_gauss").
        random_seed (int): Seed for random number generation to ensure reproducibility.
        k (int): The number of clusters to use or aim for (for semi-supervised settings).
        percent_labelled (float): The percentage of data points to be considered labelled.
        standardise (bool): If True, features will be standardized (mean=0, std=1).

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame): The loaded (and optionally standardized) DataFrame.
            - num_clusters (int): The number of clusters in the dataset.
            - plot_title (str): A title for plotting.
            - feature_columns (list): A list of column names that are features.
    """
    df = None
    num_clusters = 0
    plot_title = ""

    # Load the base dataset based on its name
    if dataset_name == "1d_simple":
        num_clusters = k
        # Assuming generate_clustering_1d_data is defined elsewhere
        df = generate_clustering_1d_data(repeat_const=100,
                                        percent_labelled=percent_labelled,
                                        random_state=random_seed)
        plot_title = dataset_name + ' (all data with histogram overlay)'

    elif dataset_name == "1d_gauss":
        num_clusters = k
        # Assuming generate_clustering_1d_gauss_anomalies is defined elsewhere
        df = generate_clustering_1d_gauss_anomalies(random_seed=random_seed,
                                                labelled_percent=percent_labelled,
                                                cluster_params=[(0, 1), (50, 3), (100, 8)],                                                    
                                                samples_per_cluster=[10000, 5000, 2500],
                                                include_anomaly_cluster=True,
                                                )
        plot_title = dataset_name + ' (all data with histogram overlay)'

    # elif dataset_name == "1d_gauss_mixed_size":
    #     df = generate_clustering_1d_gauss_anomalies(random_seed=random_seed,
                                                    
    elif dataset_name == "2d_gauss":
        num_samples = 10000
        num_clusters = k
        gauss_feature_numbers=2
        labelled_fraction = percent_labelled
        same_density = False

        # Define cluster standard deviations
        if same_density:
            std_dev = 0.6
        else:
            # Set different std deviations for each component
            # Ensure std_dev has enough elements for num_clusters
            std_dev = [0.6, 2, 0.2, 0.7, 3, 0.4, 0.6, 0.6][:num_clusters]

        # Assuming generate_clustering_2d_gauss_data is defined elsewhere
        df = generate_clustering_2d_gauss_data(n_samples=num_samples,
                                            n_components=num_clusters,
                                            num_features=gauss_feature_numbers,
                                            rand_seed=random_seed,
                                            same_density=same_density,
                                            labelled_fraction=labelled_fraction,
                                            add_anomaly_cluster=True, # Assuming this should always be True or based on param
                                            std_dev=std_dev,
                                            )
        plot_title = dataset_name + ' (all data)'

    else:
        # Assuming prepare_and_seed_dataset is defined elsewhere
        df, num_clusters = prepare_and_seed_dataset(dataset_name,
                                        label_column='class',
                                        percent_labelled=percent_labelled,
                                        k=k,
                                        random_seed=random_seed,
                                        )
        plot_title = dataset_name + ' (all data)'

    # Identify feature columns *before* potential modification or dropping of other columns
    feature_columns = [col for col in df.columns if col not in {'y_true', 'y_live'}]

    # --- Standardization Step ---
    if standardise:
        scaler = StandardScaler()
        # Apply standardization only to the identified feature columns
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
        logging.debug("Features for dataset '%s' have been standardized.", dataset_name)
    # --- End Standardization Step ---

    return df, num_clusters, plot_title, feature_columns