"""
Created on Mon Feb  6 13:05:33 2023

@author: nassirmohammad
"""
import os
import pandas as pd
import logging

# Define save_df helper
def save_df(df, filename_prefix, dataset_name, results_folder):
    filename = os.path.join(results_folder, f"{filename_prefix}_{dataset_name}.csv")
    df.to_csv(filename, index=False)
    logging.info(f"{filename_prefix.replace('_', ' ').capitalize()} saved to {filename}")

def combine_results(results_folder="results"):
    runtime_files = []
    metrics_files = []

    # Scan folder for files
    for filename in os.listdir(results_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(results_folder, filename)
            # Heuristic: runtime files only have 3 columns
            with open(filepath, 'r') as f:
                header = f.readline()
                num_columns = len(header.strip().split(","))
                if num_columns == 3 and "Runtime" in header:
                    runtime_files.append(filepath)
                elif num_columns > 3:
                    metrics_files.append(filepath)

    # Read and combine runtimes
    runtime_dfs = [pd.read_csv(file) for file in runtime_files]
    all_runtimes = pd.concat(runtime_dfs, ignore_index=True) if runtime_dfs else pd.DataFrame()

    # Read and combine metrics
    metrics_dfs = [pd.read_csv(file) for file in metrics_files]
    all_metrics = pd.concat(metrics_dfs, ignore_index=True) if metrics_dfs else pd.DataFrame()

    if all_metrics.empty:
        print("⚠️ No metrics files found.")
        return pd.DataFrame()  # return empty DataFrame

    if all_runtimes.empty:
        print("⚠️ No runtime files found. The result will only contain metrics.")

    # Merge on Algorithm and Dataset (left join metrics with runtimes)
    combined = pd.merge(all_metrics, all_runtimes, on=["Algorithm", "Dataset"], how="left") if not all_runtimes.empty else all_metrics

    # Optional: sort for readability
    combined = combined.sort_values(by=["Dataset", "Algorithm"]).reset_index(drop=True)

    print("✅ Combined results ready.")
    return combined

def escape_latex(s):
    if not isinstance(s, str):
        return s
    return (s.replace('\\', '\\textbackslash{}')
             .replace('&', '\\&')
             .replace('%', '\\%')
             .replace('$', '\\$')
             .replace('#', '\\#')
             .replace('_', '\\_')  # ← this is the one causing your error
             .replace('{', '\\{')
             .replace('}', '\\}')
             .replace('~', '\\textasciitilde{}')
             .replace('^', '\\textasciicircum{}'))

def process_df(df_cr):
    # round values
    df_cr = df_cr.round(2)

    # clean up dataset names (probably not needed anymore)
    df_cr["Dataset"] = df_cr["Dataset"].str.replace("_with_class", "", regex=False)
    df_cr["Dataset"] = df_cr["Dataset"].str.replace("_class", "", regex=False)
    df_cr["Dataset"] = df_cr["Dataset"].str.replace("_trn", "", regex=False)
    df_cr["Dataset"] = df_cr["Dataset"].str.replace("_txt", "", regex=False)

    rename_algorithms = {
        "KMeans": "k-means",
        "DBSCAN": "DBSCAN",
        "Agglomerative": "Agg",
        "GaussianMixture": "GMM",
        "ConstrainedKMeans": "C-KM",
        "SeededKMeans": "S-KM",
        "novel_method": "Ours",
        "COPKMeans": "COPKM",
    }

    # filter and rename algorithms
    methods_to_remove = {"Spectral", "MeanShift"}
    df_cr = df_cr[~df_cr["Algorithm"].isin(methods_to_remove)]
    df_cr["Algorithm"] = df_cr["Algorithm"].replace(rename_algorithms)

    # metrics to keep
    metrics = ['Purity', 'V-Measure', 'NMI', 'ARI', 'FMI', 'Runtime (s)']

    metric_dfs = {}
    for metric in metrics:
        metric_df = df_cr.pivot(index="Dataset", columns="Algorithm", values=metric)

        # Move 'Ours' to the end
        cols = list(metric_df.columns)
        if "Ours" in cols:
            cols.remove("Ours")
            cols.append("Ours")
            metric_df = metric_df[cols]

        # Escape dataset names (index)
        metric_df.index = metric_df.index.map(escape_latex)
        metric_dfs[metric] = metric_df

    return metric_dfs

def save_metric_tables_latex(metric_dfs, save_path, use_colour=False):
    os.makedirs(save_path, exist_ok=True)

    for metric, df in metric_dfs.items():
        latex_file = os.path.join(save_path, f"{metric.replace(' ', '_')}.tex")

        # Replace NaNs with '--' in all cases first
        df_copy = df.fillna('--')

        if use_colour:
            cmap = "Reds_r" if metric == "Runtime (s)" else "Greens"
            styled = df_copy.style.background_gradient(cmap=cmap, axis=1).set_caption(metric)
            with open(latex_file, "w") as f:
                f.write(styled.to_latex(hrules=True))
        else:
            df_styled = df_copy.style.format(precision=2)
            with open(latex_file, "w") as f:
                f.write(df_styled.to_latex(hrules=True))
    