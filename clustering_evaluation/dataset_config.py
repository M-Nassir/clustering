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

    # all methods run on these in reasonable time
    0: make_entry("1d_simple", 0.03, 3, plot_figure=False, standardise=False, random_seed=None),
    1: make_entry("1d_gauss", 0.2, 3, plot_figure=False, standardise=False, random_seed=None),
    2: make_entry("2d_gauss", 0.015, 8, plot_figure=False, standardise=False, random_seed=None), # 4549 6628 743 
    3: make_entry("iris_with_class", 0.2, 3, plot_figure=False, standardise=False, random_seed=None), # 8338 3480 9093
    4: make_entry("wine_with_class", 0.3, 3, plot_figure=False, standardise=False, random_seed=None), # 3169 9942
    5: make_entry("breast_cancer_class", 0.1, 2, plot_figure=False, standardise=False, random_seed=None), # 93
    6: make_entry("seeds_with_class", 0.2, 3, plot_figure=False, standardise=False, random_seed=None), # 8993
    7: make_entry("glass_with_class", 0.3, 6, plot_figure=False, standardise=False, random_seed=None), # 1986
    8: make_entry( "ionosphere_umap10_with_class", 0.1, 2, plot_figure=False, standardise=False, random_seed=None), # 4574
    9: make_entry(# good example for failure analysis as methods do not perform well
        "yeast_with_class", 0.05, 4, plot_figure=False, standardise=False, random_seed=None), 
    10: make_entry(# 21, appears more than 2 clusters, unclear ground truth
        "banknote_with_class", 0.1, 2, plot_figure=False, standardise=False, random_seed=21), 
    11: make_entry("pendigits_txt_class", 0.05, 10, plot_figure=False, standardise=False, random_seed=None),
    12: make_entry("land_mines_class", 0.3, 5, plot_figure=False, standardise=False, random_seed=None),
    13: make_entry("MNIST_UMAP10_with_class", 0.15, 10, plot_figure=False, standardise=False, random_seed=None), # 4470
    14: make_entry("6NewsgroupsUMAP10_with_class", 0.02, 10, plot_figure=False, standardise=False, random_seed=None),

    # spectral, mean shift, agglomerative and COK-KMeans too slow on these datasets; they are skipped
    15: make_entry( # highly imbalanced, one class dominates 78%, not good # 6435
        "shuttle_trn_with_class", 0.05, 3, plot_figure=False, standardise=False, random_seed=None), 
    # 16: {"name": "covtype_with_class.csv", # will require pre-processing to 10-20
    #      "percent_labelled": 0.15,
    #      "k": 7,
    #      "standardise": False,
    #      "random_seed": None,
    #     },
}

# 0 to 8 are fine - not tried on mean shift, HDBSCAN, spectral, GMM, agglomertaive 
