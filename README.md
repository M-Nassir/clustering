# A Novel Semi-Supervised Clustering Method
This project implements a semi-supervised clustering algorithm, based on the method described in the following publication:
https://arxiv.org/abs/2306.06974

The algorithm takes as input a matrix of numerical features, where each row represents an example and each column a feature. An additional column must contain partial labels (seeds), which are externally provided. Labels must be integers >= -1, with -1 reserved for unlabelled examples.

As a rule of thumb, it is recommended to provide 10-30 labelled examples per known class to ensure effective clustering.

At its core, the algorithm uses the Perception anomaly detection algorithm. Starting from the initial labelled seeds, it iteratively adds or ejects points from each cluster based on their consistency with the group. This process continues until the clusters stabilise or a maximum number of iterations is reached.

See the notebooks folder for getting started guides containing examples of using the method, and its performance against other popular clustering methods.

Key features:
- Supports both one-dimensional and multi-dimensional numerical data.
- Assigns a cluster label to every input example.
- Uses the label -1 to indicate anomalous examples.
- These anomalous examples can be reviewed in follow-up analysis, allowing users to refine labels and re-run clustering in iterative cycles.

## Installation
To install the solution via `pip`, you can use the following command:

```bash
pip install clustering_nassir
```

## Usage

```bash
pip install clustering_nassir
from clustering_nassir import SemiSupervisedClusterer

# Prepare your DataFrame
# Include feature columns plus a 'y_live' column with a few labelled examples
data = df[["feature1", "feature2", "feature3", "y_live"]].to_numpy()

# Fit the model
model = SemiSupervisedClusterer()
df["novel_method"] = model.fit(data)
```

