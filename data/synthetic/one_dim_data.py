# %% imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import seaborn as sns
import matplotlib as mpl

# %%
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
#
#                         Make the data
#
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

def generate_clustering_1d_data(repeat_const=100, percent_labelled=0.03, random_state=None):

    if random_state is not None:
        np.random.seed(random_state)

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

# %%
# df = generate_clustering_1d_data()
# plot_cluster_histograms(df)


# %%
