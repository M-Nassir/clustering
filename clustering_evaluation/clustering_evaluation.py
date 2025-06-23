# %%
# ---------------------------- Imports and Setup ----------------------------
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
import plotly.express as px
import logging

# Project root resolution
CURRENT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT_PATH = CURRENT_DIR.parent
sys.path.insert(0, str(ROOT_PATH))

# ---------------------------- Module Imports -------------------------------
from dataset_config import dataset_dict
from clustering_methods import (
    kmeans_clustering, meanshift_clustering, dbscan_clustering,
    agglomerative_clustering, gmm_clustering, spectral_clustering,
    constrained_kmeans_clustering, copk_means_clustering, hdbscan_clustering, 
    seeded_k_means_clustering, novel_clustering, dec_clustering,
    run_and_time_clusterings,
)
from utilities.plotting import (
    plot_clusters, plot_enabled_clusterings, 
    plot_confusion_matrices_for_clustering,
)
from utilities.cluster_utilities import save_df, combine_results
from utilities.evaluation_metrics import (
    compute_accuracy, compute_purity, compute_homogeneity, compute_ari,
    compute_completeness, compute_v_measure, compute_nmi, compute_fmi,
    compute_silhouette, compute_davies_bouldin, compute_calinski_harabasz,
    evaluate_clustering_metrics,
)
from utilities.generate_load_data import load_dataset
        
# %%
# -------------------------- Experiment Configuration ------------------------

IS_TESTING = True
RESULTS_FOLDER = 'results'
PLOT_FIGURES = False
SAVE_RESULTS = True
SAVE_PLOTS = False
PLOT_SAVE_PATH = Path.home() / "Google Drive/docs/A_computational_theory_of_clustering/figures"
logging.basicConfig(level=logging.DEBUG if IS_TESTING else logging.WARNING,
                    format="%(levelname)s: %(message)s"
)
# Random seed
default_random_seed = np.random.randint(0, 10000)
logging.debug("Random seed used for this run: %s", default_random_seed)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# -------------------------- Dataset Setup -----------------------------------

single_dataset_index = None
dataset_indices = [single_dataset_index] if single_dataset_index is not None else list(dataset_dict.keys())

for dataset_index in dataset_indices:     
    dataset_settings = dataset_dict[dataset_index]

    # Pull from settings with fallbacks
    default_random_seed = dataset_settings["random_seed"] if dataset_settings.get("random_seed") is not None else default_random_seed
    dataset_name = dataset_settings["name"]
    plot_figures = dataset_settings.get("plot_figure", PLOT_FIGURES)
    k = dataset_settings.get("k", None)
    percent_labelled = dataset_settings.get("percent_labelled", None)
    standardise = dataset_settings.get("standardise", True)

    logging.debug("Dataset name: %s", dataset_name)
    logging.debug("Random seed: %s", default_random_seed)

    # -------------------------- Load Dataset ------------------------------------

    logging.info(
        "Loading dataset: %s with parameters: \nrandom_seed=%s,\nk=%s,\npercent_labelled=%s,\nstandardise=%s",
        dataset_name, default_random_seed, k, percent_labelled, standardise,
    )
    df, num_clusters, plot_title, feature_columns = load_dataset(
        dataset_name,
        default_random_seed,
        k,
        percent_labelled,
        standardise,
    )

    logging.info("Dataset '%s' loaded successfully", dataset_name)

    logging.debug("Number of examples: %s", df.shape[0])
    logging.debug("Number of features: %s", df.shape[1])
    logging.debug('Class distribution:')
    logging.debug(df['y_live'].value_counts())
    logging.debug('\nClass proportions (%):')
    logging.debug(df['y_true'].value_counts(normalize=True).mul(100).round(2))

    # -------------------- Plot dataset and seeds only separately -----------------

    if PLOT_FIGURES:
        # Plot with true labels
        fig1 = plot_clusters(
            df, feature_columns, label_column='y_true',
            x_axis_label='', y_axis_label='Count',
            legend_label='Cluster labels', 
            title='', show_seeds_only=False
        )

        # Plot with seed labels only
        fig2 = plot_clusters(
            df, feature_columns, label_column='y_live', 
            title=f"{dataset_name} (seeds only)", 
            show_seeds_only=True
        )

        if SAVE_PLOTS:
            fig1_path = os.path.join(PLOT_SAVE_PATH, f"{dataset_name}_ytrue.png")
            fig2_path = os.path.join(PLOT_SAVE_PATH, f"{dataset_name}_ylive_seeds_only.png")

            fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
            fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')

    # ---------------------------- Clustering Algorithm Setup and Execution ------------------------

    # Flags to enable/disable specific clustering algorithms
    clustering_flags = {
        # Unsupervised clustering methods
        'KMeans': True,
        'MeanShift': True, #
        'DBSCAN': True,
        'HDBSCAN': True,
        'Agglomerative': True, #
        'GMM': True,
        'Spectral': True, #

        # Semi-supervised clustering methods
        'ConstrainedKMeans': True,
        'COPKMeans': True, #
        'SeededKMeans': True,
        'novel_method': True,

        # Deep learning (self-)unsupervised clustering
        'DEC': True,
    }

    # Configuration dictionary mapping method names to their functions and parameters
    clustering_configs = {
        'KMeans': {
            'function': kmeans_clustering,
            'params': {
                'n_clusters': num_clusters,
                'target_column': 'y_true',
                'remap_labels': True,
            }
        },
        'MeanShift': {
            'function': meanshift_clustering,
            'params': {
                'target_column': 'y_true',
                'remap_labels': True,
            }
        },
        'DBSCAN': {
            'function': dbscan_clustering,
            'params': {
                'target_column': 'y_true',
                'remap_labels': True,
            }
        },
        'HDBSCAN': {
            'function': hdbscan_clustering,
            'params': {
                'target_column': 'y_true',
                'min_cluster_size': 5,
                'min_samples': None,
                'remap_labels': True,
            }
        },
        'Agglomerative': {
            'function': agglomerative_clustering,
            'params': {
                'n_clusters': num_clusters,
                'target_column': 'y_true',
                'remap_labels': True,
            }
        },
        'GMM': {
            'function': gmm_clustering,
            'params': {
                'n_components': num_clusters,
                'target_column': 'y_true',
                'remap_labels': True,
            }
        },
        'Spectral': {
            'function': spectral_clustering,
            'params': {
                'n_clusters': num_clusters,
                'target_column': 'y_true',
                'remap_labels': True,
            }
        },
        'ConstrainedKMeans': {
            'function': constrained_kmeans_clustering,
            'params': {
                'n_clusters': num_clusters,
                'target_column': 'y_true',
                'size_min': 15,
                'size_max': df.shape[0],  # max size capped by dataset size
                'remap_labels': True,
            }
        },
        'COPKMeans': {
            'function': copk_means_clustering,
            'params': {
                'num_clusters': num_clusters,
                'target_column': 'y_true',
                'label_column': 'y_live',
                'remap_labels': True,
            }
        },
        'SeededKMeans': {
            'function': seeded_k_means_clustering,
            'params': {
                'n_clusters': num_clusters,
                'target_column': 'y_true',
                'seeds': 'y_live',
                'remap_labels': True,
            }
        },
        'novel_method': {
            'function': novel_clustering,
            'params': {
                'target_column': 'y_true',
                'seeds': 'y_live',
                'remap_labels': False,  # No label remapping for novel method
            }
        },
        'DEC': {
            'function': dec_clustering,
            'params': {
                'num_clusters': num_clusters,
                'pretrain_epochs': 10,      # Default: 100 (reduced here for quick runs)
                'clustering_epochs': 10,    # Default: 150
                'target_column': 'y_true',
                'remap_labels': True,
            }
        },
    }

    # ---------------------------- Run Clustering Algorithms and Measure Runtime ------------------------

    SKIP_CLUSTERING = { # these methods take too long on these datasets
        "shuttle_trn_with_class": {
            "MeanShift",
            "Agglomerative",
            "Spectral",
            "COPKMeans"
        },
    }
    # Execute clustering algorithms based on the provided configurations and flags
    df, runtime_df = run_and_time_clusterings(
        df,
        dataset_name,
        feature_columns,
        clustering_configs,
        clustering_flags,
        SKIP_CLUSTERING,
    )

    # Plot results of enabled clustering algorithms if plotting is enabled
    if PLOT_FIGURES:
        plot_enabled_clusterings(
            df,
            clustering_flags,
            feature_columns,
            plot_save_path=PLOT_SAVE_PATH,
            dataset_name=dataset_name,
            save_plots=SAVE_PLOTS,
        )

    if SAVE_RESULTS:
        logging.info("Saving clustering runtimes for dataset: %s", dataset_name)
        save_df(runtime_df, "runtime", dataset_name, RESULTS_FOLDER)
        
    # ---------------------------- Metric Evaluation ------------------------

    # Define all metrics as tuples: (metric_name, metric_function, requires_ground_truth)
    all_metrics = [
        # External metrics (require ground truth labels)
        ('Accuracy', compute_accuracy, True),
        ('Purity', compute_purity, True),
        ('Homogeneity', compute_homogeneity, True),
        ('Completeness', compute_completeness, True),
        ('V-Measure', compute_v_measure, True),  # Equivalent to NMI here
        ('NMI', compute_nmi, True),
        ('ARI', compute_ari, True),
        ('FMI', compute_fmi, True),

        # Internal metrics (do NOT require ground truth labels)
        ('Silhouette Score', compute_silhouette, False),
        ('Davies-Bouldin Index', compute_davies_bouldin, False),
        ('Calinski-Harabasz Index', compute_calinski_harabasz, False),
    ]

    # Evaluate clustering results using the specified metrics
    df_metrics = evaluate_clustering_metrics(
        df=df,
        metrics_dict=all_metrics,
        dataset_name=dataset_name,
        clustering_flags=clustering_flags,
        feature_columns=feature_columns,
    )

    # Save the metrics DataFrame and display it
    if SAVE_RESULTS:
        save_df(df_metrics, "clustering_metrics", dataset_name, results_folder=RESULTS_FOLDER)
    # display(df_metrics)

    # ---------------------------- Plot Confusion Matrices ------------------------
    # Plot confusion matrices for all enabled clustering methods
    # plot_confusion_matrices_for_clustering(
    #     df, 
    #     true_label_col='y_true', 
    #     clustering_flags=clustering_flags
    # )

    # -- run only for MNIST and 6NewsgroupsUMAP10

    if '6NewsgroupsUMAP10' in dataset_name:
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

        # Merge clustering/class columns
        for col in ['y_true', 'y_live', 'KMeans', 'novel_method']:
            if col in df.columns:
                df_vis[col] = df[col]
            else:
                raise KeyError(f"Column '{col}' not found in df")

        # Define a custom, high-contrast colour palette (20 colours)
        custom_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8',
            '#ffbb78', '#98df8a', '#c5b0d5', '#c49c94', '#f7b6d2',
            '#c7c7c7', '#dbdb8d', '#9edae5', '#393b79', '#637939'
        ]

        # Create a color map with string keys to match label column type
        color_map = {str(i): custom_colors[i % len(custom_colors)] for i in range(20)}
        color_map['-1'] = 'rgb(255,0,0)'  # Red for -1 (e.g. anomaly or outlier)

        df_vis['index'] = df_vis.index.astype(str)

        for label_col in ['y_true', 'y_live', 'KMeans', 'novel_method']:
            df_vis[label_col] = df_vis[label_col].astype(str)

            # Create ordered categories starting with '-1' if present, then ascending numbers
            unique_labels = sorted(set(df_vis[label_col]) - {'-1'}, key=int)
            ordered_categories = ['-1'] + unique_labels if '-1' in df_vis[label_col].values else unique_labels

            # Build color map only for labels in this column
            color_map = {
                lbl: ('rgb(255,0,0)' if lbl == '-1' else custom_colors[i % len(custom_colors)])
                for i, lbl in enumerate(ordered_categories)
            }

            # Define hover columns excluding current label_col but including relevant info
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

    if 'MNIST_UMAP10_with_class' in dataset_name:

        logging.debug("Running UMAP visualization for MNIST dataset...")

        # Define file path
        project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
        csv_file_path = os.path.join(project_root, "data", "processed", "MNIST_UMAP2_with_images.csv")

        # Load embeddings
        df_vis = pd.read_csv(csv_file_path)

        # Merge clustering/class columns from df into df_vis
        for col in ['y_true', 'KMeans', 'novel_method']:
            if col in df.columns:
                df_vis[col] = df[col].astype(str)
            else:
                raise KeyError(f"Column '{col}' not found in df")

        # Define a custom, high-contrast colour palette for digits 0-9 and extra labels
        custom_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8',
            '#ffbb78', '#98df8a', '#c5b0d5', '#c49c94', '#f7b6d2',
            '#c7c7c7', '#dbdb8d', '#9edae5', '#393b79', '#637939'
        ]

        df_vis['index'] = df_vis.index.astype(str)

        for label_col in ['y_true', 'KMeans', 'novel_method']:
            df_vis[label_col] = df_vis[label_col].astype(str)

            # Create ordered categories starting with '-1' if present, then ascending numbers
            unique_labels = sorted(set(df_vis[label_col]) - {'-1'}, key=int)
            ordered_categories = ['-1'] + unique_labels if '-1' in df_vis[label_col].values else unique_labels

            # Build color map only for labels actually in this column
            color_map = {}
            for i, lbl in enumerate(ordered_categories):
                if lbl == '-1':
                    color_map[lbl] = 'rgb(255,0,0)'  # red for outliers
                else:
                    color_map[lbl] = custom_colors[i % len(custom_colors)]

            # Define hover columns excluding the current label_col
            hover_cols = [c for c in ['index', 'y_true', 'KMeans', 'novel_method'] if c != label_col and c in df_vis.columns]

            fig = px.scatter(
                df_vis,
                x='UMAP_1',
                y='UMAP_2',
                color=label_col,
                color_discrete_map=color_map,
                category_orders={label_col: ordered_categories},
                hover_name=None,
                hover_data=hover_cols,  # list of columns to show on hover
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

# ---- Combine the results from all datasets into a single DataFrame ----
metrics_plus_cols_to_keep = ['Algorithm', 'Dataset', 
                            'Purity', 'V-Measure', 'NMI', 
                            'ARI', 'FMI', 'Runtime (s)']
df_cr = combine_results(RESULTS_FOLDER)
df_cr = df_cr[[col for col in metrics_plus_cols_to_keep if col in df_cr.columns]]
display(df_cr)

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
    # show_mnist_image(df_vis, index=732)


    # %%
