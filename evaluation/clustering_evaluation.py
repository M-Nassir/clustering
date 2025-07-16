# %%
# ---------------------------- Imports and Setup ----------------------------
import os
import sys
import logging
from pathlib import Path
import random
import cProfile
import pstats
import pickle
import io
import numpy as np
import pandas as pd
import plotly.express as px
from IPython.display import display
pd.set_option('display.max_rows', 200)

# Project Root Resolution and add to sys.path
CURRENT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT_PATH = CURRENT_DIR.parent
sys.path.insert(0, str(ROOT_PATH))  

# Internal Module Imports 
from evaluation.evaluation_configs import dataset_dict, clustering_configs, clustering_flags, selected_metrics, skip_clustering
from evaluation.clustering_methods import run_metrics_time_clusterings
from utilities.plotting import plot_clusters, plot_enabled_clusterings, plot_confusion_matrices_for_clustering
from utilities.cluster_utilities import (
    metrics_to_dataframe, average_metrics_dataframe, save_metric_tables_as_latex,
    median_metrics_dataframe, create_metric_tables
)
from utilities.generate_load_data import load_dataset

# %%
# -------------------------- Experiment Configuration ------------------------
class Config:
    """Centralised configuration settings."""
    PROFILE_CODE = False # Enable to profile code execution time
    IS_TESTING = False   # for producing more verbose output during development
    PLOT_FIGURES = False # Enable to plot figures for each dataset
    SAVE_RESULTS = False  # Save latex tables
    SAVE_PLOTS = False   # Save plots to disk
    RESULTS_FOLDER = Path("results") # Folder to save results
    PLOT_SAVE_PATH = Path.home() / "Google Drive/docs/A_computational_theory_of_clustering/figures"
    TABLE_SAVE_PATH = Path.home() / "Google Drive/docs/A_computational_theory_of_clustering/tables"
    RANDOM_SEED = random.randint(0, 10_000)

def setup_logging(is_testing: bool) -> logging.Logger:
    """Initialise logger with verbosity based on mode."""
    logging.basicConfig(
        level=logging.DEBUG if is_testing else logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    return logging.getLogger("clustering")

# Initialise logger
logger = setup_logging(Config.IS_TESTING)

# -------------------------- Run Experiment -----------------------------------

# Set this to a specific dataset index to run only one dataset
SINGLE_DATASET_INDEX = None
dataset_indices = [SINGLE_DATASET_INDEX] if SINGLE_DATASET_INDEX is not None else list(dataset_dict.keys())

# holder for all metrics across datasets
all_metrics = {}

for dataset_index in dataset_indices:
    dataset_cfg = dataset_dict[dataset_index]

    # Resolve dataset parameters with fallbacks; default get is None
    dataset_name = dataset_cfg["name"]
    random_seed = dataset_cfg["random_seed"] if dataset_cfg.get("random_seed") is not None else Config.RANDOM_SEED
    plot_figures_dataset_specific = dataset_cfg.get("plot_figure", Config.PLOT_FIGURES)
    k = dataset_cfg.get("k")
    percent_labelled = dataset_cfg.get("percent_labelled")
    standardise = dataset_cfg.get("standardise", False)

    # load the dataset for plotting purposes and obtaining dataset characteristics
    df, num_clusters, plot_title, feature_columns = load_dataset(
        dataset_name, random_seed, k, percent_labelled, standardise,
    )

    # Save for later use
    number_of_examples = df.shape[0]
    number_of_seeds = (df['y_live'] != -1).sum()
    number_of_features = len(feature_columns)

    logger.info(
        f"Loaded dataset '{dataset_name}' with parameters:\n"
        f"  random_seed       = {random_seed}\n"
        f"  k                 = {k}\n"
        f"  percent_labelled  = {percent_labelled}\n"
        f"  standardise       = {standardise}\n"
        f"  Number of seeds   = {number_of_seeds}\n"
        f"  Number of examples= {number_of_examples}\n"
        f"  Number of features= {number_of_features}"
    )

    logger.debug("Class distribution (y_live):\n%s", df['y_live'].value_counts())

    #  -------------------- Plot Dataset and Seeds Separately --------------------

    if Config.PLOT_FIGURES:
        logger.info("Plotting dataset: %s", dataset_name)

        # Plot with true labels
        fig1 = plot_clusters(
            df,
            feature_columns,
            label_column='y_true',
            x_axis_label='',
            y_axis_label='Count',
            legend_label='Cluster Labels',
            title=f"{dataset_name} (Ground Truth)",
            show_seeds_only=False,
        )

        # Plot with seed labels only
        fig2 = plot_clusters(
            df,
            feature_columns,
            label_column='y_live',
            title=f"{dataset_name} (Seed Labels Only)",
            show_seeds_only=True,
        )

        if Config.SAVE_PLOTS:
            Config.PLOT_SAVE_PATH.mkdir(parents=True, exist_ok=True)

            fig1_path = Config.PLOT_SAVE_PATH / f"{dataset_name}_ytrue.png"
            fig2_path = Config.PLOT_SAVE_PATH / f"{dataset_name}_ylive_seeds_only.png"

            fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
            fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')

            logger.info("Saved plots to:\n- %s\n- %s", fig1_path, fig2_path)

    # -------------------- # Execute clustering algorithms based on the provided configurations and flags --------------------
    logger.info("******* Preparing to apply clustering methods for dataset %s *******" % dataset_name)

    if Config.PROFILE_CODE:
        pr = cProfile.Profile()
        pr.enable()
        
    metrics_df, df_one_result = run_metrics_time_clusterings(
        dataset_name = dataset_name,
        random_seed = Config.RANDOM_SEED,
        k = k,
        percent_labelled = percent_labelled,
        standardise = standardise,
        clustering_configs = clustering_configs,
        clustering_flags = clustering_flags,
        skip_clusterings = skip_clustering,
        num_repeats=1,
        load_dataset=load_dataset,
        selected_metrics=selected_metrics,
        num_examples = number_of_examples,
    )
    
    all_metrics[dataset_name] = metrics_df

    if Config.PROFILE_CODE:
        pr.disable()

        # Create a stream to hold profiling stats
        s = io.StringIO()
        sortby = 'cumtime'  # Sort by cumulative time to see bottlenecks

        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(50)  # Show top 50 lines
        print(s.getvalue())

    logger.info("******* Completed running clustering and metrics for dataset %s for all methods *******" % dataset_name)

    # Plot results of enabled clustering algorithms if plotting is enabled
    if Config.IS_TESTING:
        if plot_figures_dataset_specific: # this tells us if in config file we want to plot for this dataset
            plot_enabled_clusterings(
                df,
                clustering_flags,
                feature_columns,
                plot_save_path=Config.PLOT_SAVE_PATH,
                dataset_name=dataset_name,
                save_plots=Config.SAVE_PLOTS,
            )

    # ---------------------------- Plot Confusion Matrices ------------------------
    # Plot confusion matrices for all enabled clustering methods
    if Config.IS_TESTING:
        plot_confusion_matrices_for_clustering(
            df, 
            true_label_col='y_true', 
            clustering_flags=clustering_flags
        )

    # -- run only for MNIST and 6NewsgroupsUMAP10

    if dataset_name == '6NewsgroupsUMAP10':

        # Define file path
        project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
        csv_file_path = os.path.join(project_root, "data", "processed", "6NewsgroupsUMAP2_embeddings.csv")

        # Load embeddings
        df_vis = pd.read_csv(csv_file_path)

        def format_email_body(text, words_per_line=10):
            words = str(text).split()
            lines = [' '.join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
            return '<br>'.join(lines)

        # Create a formatted email body column with line breaks every 10 words for hover display
        df_vis['email_body_formatted'] = df_vis['email_body'].apply(format_email_body)

        # Assign clustering results and labels from df_one_result dict
        for col in ['KMeans', 'novel_method']:
            if col in df_one_result:
                # Assign y_true and y_live only once if not present
                if 'y_true' not in df_vis:
                    df_vis['y_true'] = df_one_result[col]['y_true']
                if 'y_live' not in df_vis:
                    df_vis['y_live'] = df_one_result[col]['y_live']

                # Assign clustering results for this method
                df_vis[col] = df_one_result[col][col]
            else:
                raise KeyError(f"Column '{col}' not found in df_one_result")

        # Define a custom, high-contrast colour palette (20 colours)
        custom_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8',
            '#ffbb78', '#98df8a', '#c5b0d5', '#c49c94', '#f7b6d2',
            '#c7c7c7', '#dbdb8d', '#9edae5', '#393b79', '#637939'
        ]

        # Build global color map across all label columns for consistency
        label_columns = ['y_true', 'y_live', 'KMeans', 'novel_method']
        all_labels = set()

        for col in label_columns:
            if col in df_vis:
                all_labels.update(df_vis[col].astype(str).unique())

        # Make sure '-1' (anomalies) is first and red
        all_labels.discard('-1')
        ordered_labels = ['-1'] + sorted(all_labels, key=int)

        global_color_map = {
            lbl: ('rgb(255,0,0)' if lbl == '-1' else custom_colors[i % len(custom_colors)])
            for i, lbl in enumerate(ordered_labels)
        }

        df_vis['index'] = df_vis.index.astype(str)

        for label_col in label_columns:
            if label_col not in df_vis:
                continue  # skip if column missing

            df_vis[label_col] = df_vis[label_col].astype(str)

            # Only keep labels present in this column, preserving order from global
            ordered_categories = [lbl for lbl in ordered_labels if lbl in df_vis[label_col].values]

            # Use subset of global color map for this column
            color_map = {lbl: global_color_map[lbl] for lbl in ordered_categories}

            # Define hover columns, excluding current label_col
            hover_cols = [
                col for col in ['index', 'y_true', 'y_live', 'KMeans', 'novel_method', 'category', 'top_keywords', 'email_body_formatted']
                if col != label_col and col in df_vis.columns
            ]

            fig = px.scatter(
                df_vis,
                x='UMAP_1',
                y='UMAP_2',
                color=label_col,
                color_discrete_map=color_map,
                category_orders={label_col: ordered_categories},
                hover_name=None,
                hover_data=hover_cols,
                title=f'UMAP projection colored by {label_col}',
                width=1400,
                height=900,
            )

            # Add white edges to markers
            for trace in fig.data:
                trace.marker.line.color = 'white'
                trace.marker.line.width = 1.5

            fig.update_layout(
                legend_title_text=label_col,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    title='', showgrid=False, showline=True,
                    linecolor='black', linewidth=1,
                    zeroline=False, ticks='outside'
                ),
                yaxis=dict(
                    title='', showgrid=False, showline=True,
                    linecolor='black', linewidth=1,
                    zeroline=False, scaleanchor="x",
                    scaleratio=1, ticks='outside'
                )
            )

            fig.show()

    # -- run only for MNIST

    if dataset_name == 'MNIST_UMAP10':
        logging.debug("Running UMAP visualization for MNIST dataset...")

        # Define save directory
        os.makedirs(Config.PLOT_SAVE_PATH, exist_ok=True)

        # Define file path
        project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
        csv_file_path = os.path.join(project_root, "data", "processed", "MNIST_UMAP2_with_images.csv")

        # Load embeddings
        df_vis = pd.read_csv(csv_file_path)

        # Assign clustering results and labels from df_one_result dict
        for col in ['KMeans', 'novel_method']:
            if col in df_one_result:
                # Assign y_true and y_live only once if not present
                if 'y_true' not in df_vis:
                    df_vis['y_true'] = df_one_result[col]['y_true']
                if 'y_live' not in df_vis:
                    df_vis['y_live'] = df_one_result[col]['y_live']

                # Assign clustering results for this method
                df_vis[col] = df_one_result[col][col]
            else:
                raise KeyError(f"Column '{col}' not found in df_one_result")

        df_vis['index'] = df_vis.index.astype(str)

        # Collect global labels
        all_labels = set()
        for label_col in ['y_true', 'y_live', 'KMeans', 'novel_method']:
            all_labels.update(df_vis[label_col].unique())

        all_labels = {str(l) for l in all_labels}
        unique_labels = sorted([l for l in all_labels if l != '-1'], key=int)
        ordered_categories = ['-1'] + unique_labels if '-1' in all_labels else unique_labels

        # Custom colours
        custom_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8',
            '#ffbb78', '#98df8a', '#c5b0d5', '#c49c94', '#f7b6d2',
            '#c7c7c7', '#dbdb8d', '#9edae5', '#393b79', '#637939'
        ]

        custom_colors = [
            'green', 'blue', 'black',
            'orange', 'purple', 'brown',
            'pink', 'cyan', 'darkblue',
            'violet', 'magenta', 'black',
        ]

        global_color_map = {}
        for i, lbl in enumerate(ordered_categories):
            global_color_map[lbl] = 'rgb(255,0,0)' if lbl == '-1' else custom_colors[i % len(custom_colors)]

        # Create and save plots
        for label_col in ['y_true', 'y_live', 'KMeans', 'novel_method']:
            df_vis[label_col] = df_vis[label_col].astype(str)

            hover_cols = [c for c in ['index', 'y_true', 'y_live', 'KMeans', 'novel_method'] if c != label_col and c in df_vis.columns]

            fig = px.scatter(
                df_vis,
                x='UMAP_1',
                y='UMAP_2',
                color=label_col,
                color_discrete_map=global_color_map,
                category_orders={label_col: ordered_categories},
                hover_name=None,
                hover_data=hover_cols,
                title=f'UMAP projection colored by {label_col}',
                width=1400,
                height=900,
            )

            # Add white borders
            for trace in fig.data:
                trace.marker.line.color = 'white'
                trace.marker.line.width = 1.5

            fig.update_layout(
                legend_title_text=label_col,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    title='', showgrid=False, showline=True,
                    linecolor='black', linewidth=1,
                    zeroline=False, ticks='outside'
                ),
                yaxis=dict(
                    title='', showgrid=False, showline=True,
                    linecolor='black', linewidth=1,
                    zeroline=False, scaleanchor="x",
                    scaleratio=1, ticks='outside'
                )
            )

            fig.show()

            # Save to file
            save_file = os.path.join(Config.PLOT_SAVE_PATH, f"mnist_umap_{label_col}.png")
            fig.write_image(save_file, scale=2)  # scale=2 improves resolution

# %% -------------------- Convert all metrics to a DataFrame for easier analysis --------------------
def display_and_save_metric_table(metric: str, df: pd.DataFrame):
    """Display a styled metric table and optionally save it as CSV."""
    cmap = "Reds_r" if "runtime (s)" in metric.lower() else "Greens"
    formatter = lambda x: "--" if pd.isna(x) else f"{x:.2f}"

    styled = (
        df.style
        .background_gradient(cmap=cmap, axis=1)
        .format(formatter)
        .set_caption(metric)
    )
    display(styled)

    if Config.SAVE_RESULTS:
        # Safe filename: lowercase, no spaces/parentheses
        safe_filename = (
            metric.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        ) + ".csv"
        df.to_csv(os.path.join(Config.RESULTS_FOLDER, safe_filename), index=True)
        
# %% -------------------- Convert all metrics to a DataFrame for easier analysis --------------------
metric_tables = {}
df_metrics = metrics_to_dataframe(all_metrics)
df_metrics["value"] = df_metrics["value"].round(2)

# Use median metrics
df_median_metrics = median_metrics_dataframe(df_metrics)
metric_tables = create_metric_tables(df_median_metrics)

if Config.SAVE_RESULTS:
    save_metric_tables_as_latex(metric_tables, Config.TABLE_SAVE_PATH)

# Loop through and display/save
for metric, df in metric_tables.items():
    display_and_save_metric_table(metric, df)

# %% -------------------- Optionally save entire metric_tables dict --------------------
if Config.SAVE_RESULTS:
    with open(os.path.join(Config.RESULTS_FOLDER, "metric_tables.pkl"), "wb") as f:
        pickle.dump(metric_tables, f)






# %% --------------------------------------------------
#
#                         WIP code 
#
#    --------------------------------------------------
# %% process average values for each metric

# # Specify the metrics to keep (case-insensitive match)
# metrics_to_keep = {"NMI", "V-MEASURE", "PURITY", "ARI", "FMI"}

# # Initialise the result dictionary
# average_ranks = {}

# # Loop through all CSV files
# for file in Config.RESULTS_FOLDER.glob("*.csv"):
#     metric_name = file.stem.upper()  # Normalize to uppercase

#     if metric_name not in metrics_to_keep:
#         continue  # Skip unwanted metrics

#     df = pd.read_csv(file, index_col=0)

#     # Find the Average Rank row (case-insensitive, strip whitespace)
#     avg_row = df.loc[df.index.str.strip().str.lower() == "average rank"]
#     if not avg_row.empty:
#         average_ranks[metric_name] = avg_row.iloc[0]

# # Combine into a DataFrame
# df_average_ranks = pd.DataFrame(average_ranks).T  # Metrics as rows

# # Add an overall average row across the selected metrics
# df_average_ranks.loc["MEAN RANK"] = df_average_ranks.mean(axis=0, numeric_only=True)

# # Display or save
# print(df_average_ranks.round(2))
# # df_average_ranks.to_csv("average_ranks_summary_filtered.csv")


    
    # %% Experiment code when exploring results for MNIST
    # import matplotlib.pyplot as plt
    # import numpy as np

    # def show_mnist_image(df_vis, index, img_size=(8, 8), thumbnail_size=(2, 2)):
    #     """
    #     Display the MNIST digit image from df_vis at the specified index as a thumbnail.

    #     Parameters:
    #     - df_vis: DataFrame containing the 'image_pixels' column
    #     - index: integer row index in df_vis to display
    #     - img_size: tuple, the shape of the image (default 8x8)
    #     - thumbnail_size: tuple, size of the figure in inches (default 2x2)
    #     """
    #     # Read pixel string
    #     pixels_str = df_vis.loc[index, 'image_pixels']
        
    #     # Convert string representation of list to numpy array
    #     pixels = np.array(eval(pixels_str), dtype=float)
        
    #     # Rescale pixel values (original MNIST 8x8 grayscale scaled 0-15)
    #     pixels = (pixels / pixels.max()) * 255
    #     pixels = pixels.astype(np.uint8)
        
    #     # Reshape to image size
    #     img = pixels.reshape(img_size)
        
    #     # Plot the image with thumbnail size
    #     plt.figure(figsize=thumbnail_size)
    #     plt.imshow(img, cmap='gray', interpolation='nearest')
    #     plt.axis('off')
    #     plt.title(f"MNIST digit at index {index}", fontsize=8)
    #     plt.show()

    #     print("Index number:", index)

    # # Example usage
    # show_mnist_image(df_vis, index=150)

# %%
# import pandas as pd

# data = [
#     ["1d_gauss",           17528,    1,   3,   0.2,   12],
#     ["2d_gauss",           10300,    2,   8,   1,     13],
#     ["6Newsgroups_UMAP10", 10496,   10,   6,   1,     17],
#     ["MNIST_UMAP10",        1797,   10,  10,   5,      9],
#     ["banknote",            1372,    4,   2,   2,     13],
#     ["breast_cancer",        569,   30,   2,   7,     20],
#     # ["cover_type",       581012,   54,   7,   0.02,  15],
#     ["glass",               214,     9,   6,  30,     10],
#     ["ionosphere_UMAP10",   351,    10,   2,  10,     17],
#     ["iris",                150,     4,   3,  20,     10],
#     ["land_mines",          338,     3,   5,  30,     20],
#     ["pendigits",         10992,    16,  10,  2.5,    27],
#     ["seeds",               210,     7,   3,  20,     14],
#     ["shuttle",           43500,     9,   3,  0.2,    29],
#     ["wine",                178,    13,   3,  30,     17],
#     ["yeast",              1484,     8,   4,   5,     18],
# ]

# columns = ["dataset", "n", "n_features", "n_classes", "label_fraction (%)", "points_per_class"]

# df = pd.DataFrame(data, columns=columns)

# # Optional: show the DataFrame
# print(df)

# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming df is your dataframe with columns such as:
# # ['dataset', 'n_samples', 'n_features', 'n_classes', 'min_samples_per_class', 'max_samples_per_class']

# # Drop the dataset name column since it's categorical
# df_numeric = df.drop(columns=['dataset'])

# # Create pairplot to visualize scatter plots and distributions of all numeric columns
# sns.pairplot(df_numeric)

# plt.suptitle("Pairwise scatter plots of dataset characteristics", y=1.02)
# plt.show()


# ranking:

# if Config.SAVE_RESULTS:
#     df_metrics = metrics_to_dataframe(all_metrics)
#     df_metrics["value"] = df_metrics["value"].round(2)

#     # median metrics
#     df_median_metrics = median_metrics_dataframe(df_metrics)
#     metric_tables = create_metric_tables_and_save_tex(df_median_metrics, Config.TABLE_SAVE_PATH)

#     # average metrics
#     # df_avg_metrics = average_metrics_dataframe(df_metrics)
#     # metric_tables = create_metric_tables_and_save_tex(df_avg_metrics, Config.TABLE_SAVE_PATH)

#     # New dictionary to hold ranked tables
#     ranked_metric_tables = {}
    
#     for metric, df in metric_tables.items():
#         # Ensure index name is set
#         df.index.name = 'dataset'

#         # Compute ranks
#         ranks = df.rank(ascending=False, axis=1, method='average', na_option='keep')
#         avg_rank = ranks.mean(axis=0)

#         # Create a DataFrame with average rank row
#         avg_rank_df = pd.DataFrame([avg_rank], index=pd.Index(['Average Rank'], name='dataset'))

#         # Append average rank to original DataFrame
#         df_with_rank = pd.concat([df, avg_rank_df])

#         # Store in new dictionary
#         ranked_metric_tables[metric] = df_with_rank

#         # Display for verification
#         display(df_with_rank.style.format("{:.2f}", na_rep="--"))

#         # Optional: Save LaTeX (commented out)
#         # tex_path = Config.TABLE_SAVE_PATH / f"{metric}.tex"
#         # df_with_rank.to_latex(tex_path, index=True, float_format="%.2f", na_rep="--")
#         # logger.info("Saved LaTeX table with ranks for metric '%s' to %s", metric, tex_path)