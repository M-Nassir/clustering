# Semi-Supervised Clustering
This project implements a semi-supervised clustering algorithm, based on the method described in the following publication:
https://arxiv.org/abs/2306.06974

The algorithm takes as input a matrix of numerical features, where each row represents an example and each column a feature. An additional column must contain partial labels (seeds), which are externally provided. Labels must be integers, with -1 reserved for unlabelled examples.

It is recommended to provide at least 10 labelled examples per class to ensure effective clustering.

At its core, the algorithm uses the Perception anomaly detection algorithm. Starting from the initial labelled seeds, it iteratively adds or ejects points from each cluster based on their consistency with the group. This process continues until the clusters stabilise.

Key features:
Supports both one-dimensional and multi-dimensional numerical data.
Assigns a cluster label to every input example.
Uses the label -1 to indicate anomalous or unassigned examples.
These anomalous examples can be reviewed in follow-up analysis, allowing users to refine labels and re-run clustering in iterative cycles.

