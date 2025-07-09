import numpy as np

def make_entry(name, percent_labelled, k, plot_figure=False, standardise=False, random_seed=None):
    return {
        "name": name,
        "percent_labelled": percent_labelled,
        "k": k,
        "plot_figure": plot_figure,
        "standardise": standardise,
        "random_seed": random_seed,
    }

# %% read in dataset; the following datasets have been pre-processed so that the last column
# is the class label, and the rest are features, the class column is integer encoded,
# all feature columns have been given a name. 

dataset_dict = {
    # 0: make_entry("1d_simple", 0.03, 3, plot_figure=False, standardise=False, random_seed=None),
    1: make_entry("1d_gauss", 0.005, 3, plot_figure=False, standardise=False, random_seed=None),
    2: make_entry("2d_gauss", 0.01, 6, plot_figure=True, standardise=False, random_seed=8858), # 4549 6628 743 8858
    3: make_entry("iris", 0.2, 3, plot_figure=False, standardise=False, random_seed=9093), # 8338 3480 9093
    4: make_entry("wine", 0.3, 3, plot_figure=False, standardise=False, random_seed=9942), # 3169 9942
    5: make_entry("breast_cancer", 0.1, 2, plot_figure=False, standardise=False, random_seed=1451), # 1451
    6: make_entry("seeds", 0.2, 3, plot_figure=False, standardise=False, random_seed=8993), # 8993
    7: make_entry("glass", 0.3, 6, plot_figure=False, standardise=False, random_seed=1986), # 1986
    8: make_entry( "ionosphere_UMAP10", 0.1, 2, plot_figure=False, standardise=False, random_seed=4574), # 4574
    9: make_entry(# good example for failure analysis as methods do not perform well
        "yeast", 0.05, 4, plot_figure=False, standardise=False, random_seed=None), 
    10: make_entry(# 21, appears more than 2 clusters, unclear ground truth
        "banknote", 0.02, 2, plot_figure=False, standardise=False, random_seed=21), # 21
    11: make_entry("pendigits", 0.05, 10, plot_figure=False, standardise=False, random_seed=769), # 769 
    12: make_entry("land_mines", 0.3, 5, plot_figure=False, standardise=False, random_seed=None),
    13: make_entry("MNIST_UMAP10", 0.05, 10, plot_figure=False, standardise=False, random_seed=None), # 4470
    14: make_entry("6NewsgroupsUMAP10", 0.02, 10, plot_figure=False, standardise=False, random_seed=None),
    15: make_entry( # highly imbalanced, one class dominates 78%, not good # 6435
        "shuttle", 0.01, 3, plot_figure=False, standardise=False, random_seed=2196), #2196 
    16: make_entry("cover_type", percent_labelled=0.01, k=7, plot_figure=False, standardise=False, random_seed=9493),
}
