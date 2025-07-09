# %% Imports
import os
import pandas as pd

""" We use the following datasets:
- Iris Data: n: 150, d: 4, classes: 3
             A classic and very clean dataset. 
             It's small, easy to visualize, and has relatively well-separated clusters, 
             making it ideal for initial testing and demonstrating basic algorithm functionality.
- Wine Data: n: 178, d: 13, classes: 3
             Another popular choice for its clear cluster structure and slightly higher 
             dimensionality than Iris, providing a good step up in complexity.
- Breast Cancer Data: n: 569, d: 30, classes: 2
             A real-world medical dataset. While binary, it's often used to test how well clustering 
             algorithms can separate these two important groups based on various diagnostic features.
- Seed Data: n: 210, d: 7, classes: 3
             A clean and straightforward dataset with clear cluster definitions, making it suitable 
             for comparing the core performance of clustering algorithms.
- Glass Data: n: 214, d: 9, classes: 6
             Offers a higher number of classes, which can test an algorithm's ability to distinguish 
             more granular groups. Some classes might be less separable, providing a moderate challenge.
- Ionosphere Data: n: 351, d: 34, classes: 2
             A more complex dataset with a higher number of features and a binary classification. As one 
             class is good, and the other is bad, the bad class may not form a cohere cluster.
- Shuttle Data: n: 49097, d: 9, classes: 7
             A large dataset with a significant number of samples and features, With 58,000 instances, 
             it serves as an excellent benchmark for assessing the computational efficiency and 
             scalability of clustering algorithms. Algorithms that perform well on smaller datasets 
             might struggle or be very slow on this larger volume of data.
             Class 5, 5% of data is all predicted as anomalies, what is an anomaly here, is the small
             group of instances another group or an anomoalous group? Ground truth can become blurry here.
- Yeast Data: n: 1484, d: 8, classes: 10
             A larger dataset with more classes, allowing for evaluation on more complex and potentially 
             imbalanced clustering scenarios. # good example for failure analysis as methods do not perform well?
- Banknotes Data: n: 1372, d: 5, classes: 2
            Appears that there are more than 2 clusters, ground truth not clear, 5 clusters? Not used further.
- Pendigits Data: n: 7494, d: 16, classes: 10
             A larger dataset with more classes, allowing for evaluation on more complex and potentially
             imbalanced clustering scenarios. The dataset consists of handwritten digits, which can be
             challenging for clustering algorithms due to the variability in handwriting styles.
- CovType Data: n: 581012, d: 54, classes: 7
                A very large dataset with a high number of features and classes, suitable for testing the
                scalability and performance of clustering algorithms. The dataset is large and complex,
                making it suitable for benchmarking algorithms on high-dimensional data.
- MNIST Data: n: 60000, d: 784, classes: 10
             A classic dataset for handwritten digit recognition, with a large number of samples and high
             dimensionality. It serves as a benchmark for clustering algorithms, especially in terms of
             scalability and performance on high-dimensional data.
             The dataset is large and complex, making it suitable for testing the scalability and efficiency
             of clustering algorithms. The high dimensionality and variability in digit representation
             can challenge clustering methods, especially those that rely on distance metrics.
-20Newsgroups Data: n: 11314, d: 1000, classes: 20
             A text dataset with 20 different newsgroups, providing a challenge for clustering algorithms
             due to the high dimensionality and the need for effective text representation.
- Fashion MNIST Data: n: 60000, d: 784, classes: 10
             A dataset of fashion items, similar in structure to MNIST but with different classes. It provides
             a more complex clustering challenge due to the variety of clothing items and their features.
             The dataset is large and complex, making it suitable for testing the scalability and efficiency
             of clustering algorithms. The high dimensionality and variability in clothing item representation
             can challenge clustering methods, especially those that rely on distance metrics.
"""

# %% Paths
# Get project root (assumes script is in a subfolder like 'utilities')
CURRENT_DIR = os.getcwd()
ROOT_PATH = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
DATA_DIR = os.path.join(ROOT_PATH, "data", 'raw', "tabular")
SAVE_DATA_DIR = os.path.join(ROOT_PATH, "data", 'processed')

# %% Function: Load and Rename Class Column
def load_and_rename_class_column(filename: str, class_column: str) -> pd.DataFrame:
    """
    Load a CSV file from data/tabular and rename a given class column to 'class'.

    Parameters:
    - filename (str): Name of the CSV file (e.g., "Seed_Data.csv")
    - class_column (str): Column to be renamed to 'class'

    Returns:
    - pd.DataFrame: DataFrame with standardised 'class' column
    """
    csv_path = os.path.join(DATA_DIR, filename)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    if class_column not in df.columns:
        raise ValueError(f"Column '{class_column}' not found. Available columns: {df.columns.tolist()}")

    df = df.rename(columns={class_column: 'class'})
    return df

# %% ------------- process the shuttle dataset -------------
filename = "shuttle_trn.csv"
path_to_file = os.path.join(DATA_DIR, filename)

# Load the dataset (using whitespace delimiter)
df = pd.read_csv(path_to_file, header=None, sep=r'\s+')

# Assign column names: f1, f2, ..., class
num_columns = df.shape[1]
column_names = [f"f{i+1}" for i in range(num_columns - 1)] + ['original_class']
df.columns = column_names

# Add another column 'class' where rare classes are labeled as -1 (outliers), others stay the same
outlier_labels = {2, 3, 6, 7}
df['class'] = df['original_class'].apply(lambda x: -1 if x in outlier_labels else x)

df.drop(columns=['original_class'], inplace=True)  
print(df.head())

# save the modified DataFrame to a new CSV file
output_filename = "shuttle_trn_with_class.csv"
output_path = os.path.join(SAVE_DATA_DIR, output_filename)
df.to_csv(output_path, index=False)

# %% ------------- process the iris dataset -------------
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Create DataFrame with feature columns
df_iris = pd.DataFrame(X, columns=[f"f{i+1}" for i in range(X.shape[1])])
df_iris['class'] = y

# Print the first few rows to verify
print(df_iris.head())

# Save the processed dataset
output_filename = "iris_with_class.csv"
output_path = os.path.join(SAVE_DATA_DIR, output_filename)
df_iris.to_csv(output_path, index=False)


# %% ------------- process the ionosphere dataset -------------

filename = "ionosphere.csv"
path_to_file = os.path.join(DATA_DIR, filename)

# Load the dataset (assumes no header)
df = pd.read_csv(path_to_file, header=None)

# Assign feature column names: f1, f2, ..., class
num_columns = df.shape[1]
column_names = [f"f{i+1}" for i in range(num_columns - 1)] + ['class']
df.columns = column_names

# Map class labels: 'g' → 1 (good), 'b' → 0 (bad)
df['class'] = df['class'].map({'g': 1, 'b': 0})

# Sanity check: ensure only two values remain
assert df['class'].nunique() == 2, "Unexpected number of unique class values"

# Print to verify
print(df.head())

# Save the processed dataset
output_filename = "ionosphere_with_class.csv"
output_path = os.path.join(SAVE_DATA_DIR, output_filename)
df.to_csv(output_path, index=False)



# %% ------------- process the ionosphere dataset using UMAP -------------
import os
import pandas as pd
import umap

filename = "ionosphere.csv"
path_to_file = os.path.join(DATA_DIR, filename)

# Load the dataset (assumes no header)
df = pd.read_csv(path_to_file, header=None)

# Assign feature column names: f1, f2, ..., class
num_columns = df.shape[1]
column_names = [f"f{i+1}" for i in range(num_columns - 1)] + ['class']
df.columns = column_names

# Map class labels: 'g' → 1 (good), 'b' → 0 (bad)
df['class'] = df['class'].map({'g': 1, 'b': 0})

# Sanity check: ensure only two values remain
assert df['class'].nunique() == 2, "Unexpected number of unique class values"

# Separate features and labels
X = df.drop(columns=['class']).values
y = df['class'].values

# Apply UMAP to reduce to 10 dimensions
reducer = umap.UMAP(n_components=10, random_state=42)
X_umap = reducer.fit_transform(X)

# Create a new DataFrame with reduced features and class label
umap_columns = [f"umap_dim_{i+1}" for i in range(10)]
df_umap = pd.DataFrame(X_umap, columns=umap_columns)
df_umap['class'] = y

# Print first few rows to verify
print(df_umap.head())

# Save the UMAP-processed dataset
output_filename = "ionosphere_umap10_with_class.csv"
output_path = os.path.join(SAVE_DATA_DIR, output_filename)
df_umap.to_csv(output_path, index=False)


# %% ------------- process the breast cancer dataset -------------
import os
import pandas as pd

filename = "breast_cancer.csv"
path_to_file = os.path.join(DATA_DIR, filename)

# Load the original CSV
df = pd.read_csv(path_to_file)

# Drop any unnamed columns caused by trailing commas
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Drop the 'id' column
df.drop(columns=["id"], inplace=True)

# Convert 'diagnosis' to numeric (M = 1, B = 0)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Rename 'diagnosis' column to 'class'
df.rename(columns={"diagnosis": "class"}, inplace=True)

# Move the 'class' column to the end
columns = list(df.columns)
columns.remove("class")
columns.append("class")
df = df[columns]

# Save the processed dataset
output_filename = "breast_cancer_class.csv"
output_path = os.path.join(SAVE_DATA_DIR, output_filename)
df.to_csv(output_path, index=False)

# %% UMAP processing for breast cancer dataset
import os
import pandas as pd
import umap

filename = "breast_cancer.csv"
path_to_file = os.path.join(DATA_DIR, filename)

# Load the original CSV
df = pd.read_csv(path_to_file)

# Drop any unnamed columns caused by trailing commas
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Drop the 'id' column
df.drop(columns=["id"], inplace=True)

# Convert 'diagnosis' to numeric (M = 1, B = 0)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Rename 'diagnosis' column to 'class'
df.rename(columns={"diagnosis": "class"}, inplace=True)

# Move the 'class' column to the end
columns = list(df.columns)
columns.remove("class")
columns.append("class")
df = df[columns]

# Separate features and labels
X = df.drop(columns=['class']).values
y = df['class'].values

# Apply UMAP to reduce to 10 dimensions
reducer = umap.UMAP(n_components=20, random_state=42)
X_umap = reducer.fit_transform(X)

# Create a new DataFrame with reduced features and class label
umap_columns = [f"umap_dim_{i+1}" for i in range(20)]
df_umap = pd.DataFrame(X_umap, columns=umap_columns)
df_umap['class'] = y

# Save the UMAP-processed dataset
output_filename = "breast_cancer_umap20_class.csv"
output_path = os.path.join(SAVE_DATA_DIR, output_filename)
df_umap.to_csv(output_path, index=False)

# %% ------------- process the wine dataset -------------
filename = "wine.csv"
path_to_file = os.path.join(DATA_DIR, filename)
# Load the dataset (assumes no header)
df = pd.read_csv(path_to_file, header=None)

df['class'] = df[0]  # Assign the first column as 'class'
df.drop(columns=[0], inplace=True)  # Drop the first column
# Rename feature columns: f1, f2, ..., f13
num_features = df.shape[1] - 1  # Exclude the 'class' column
df.columns = [f"f{i+1}" for i in range(num_features)] + ['class']
# Print the first few rows to verify
print(df.head())
# Save the processed dataset
output_filename = "wine_with_class.csv"
output_path = os.path.join(SAVE_DATA_DIR, output_filename)
df.to_csv(output_path, index=False)

# %% ------------- process the glass dataset -------------
filename = "glass.csv"
path_to_file = os.path.join(DATA_DIR, filename)
# Load the dataset (assumes no header)
df = pd.read_csv(path_to_file, header=None)

# drop the first column (index) and assign the last column as 'class'
df.drop(columns=[0], inplace=True)  # Drop the first column
df['class'] = df[df.columns[-1]]  # Assign the last column as 'class'
df.drop(columns=[df.columns[-1]], inplace=True)  # Drop the last column
# Rename feature columns: f1, f2, ..., f9
num_features = df.shape[1] - 1  # Exclude the 'class' column
df.columns = [f"f{i+1}" for i in range(num_features)] + ['class']
# Print the first few rows to verify
print(df.head())
# Save the processed dataset
output_filename = "glass_with_class.csv"
output_path = os.path.join(SAVE_DATA_DIR, output_filename)
df.to_csv(output_path, index=False)

# %% ------------- process the yeast dataset -------------
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

filename = "yeast.csv"
path_to_file = os.path.join(DATA_DIR, filename)

# Load the dataset (whitespace-separated, no header)
df = pd.read_csv(path_to_file, header=None, delim_whitespace=True)

# Drop the first column (identifier like 'ADT1_YEAST')
df.drop(columns=[0], inplace=True)

# Rename the last column as 'class'
df.columns = list(df.columns[:-1]) + ['class']

# Encode the 'class' column numerically
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])  # Overwrite with numerical values

# Rename feature columns (e.g., f1, f2, ..., f8) if needed
num_features = df.shape[1] - 1  # all except 'class'
df.columns = [f"f{i+1}" for i in range(num_features)] + ['class']

# Relabel specific small classes as -1
small_classes = [1, 2, 3, 4, 8, 9]
df.loc[df['class'].isin(small_classes), 'class'] = -1

# Save to CSV
output_filename = "yeast_with_class.csv"
output_path = os.path.join(SAVE_DATA_DIR, output_filename)
df.to_csv(output_path, index=False)


# %% process covtype dataset

import os
import pandas as pd
import umap

# Paths and filenames (set your DATA_DIR and SAVE_DATA_DIR accordingly)
filename = "covtype.csv"
path_to_file = os.path.join(DATA_DIR, filename)

# Load the Covertype CSV
df = pd.read_csv(path_to_file)

# Drop any unnamed columns caused by trailing commas
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# The Covertype dataset’s label column is typically called 'Cover_Type'
# Check and rename if necessary, then move it to the end for convenience
if 'Cover_Type' in df.columns:
    df.rename(columns={'Cover_Type': 'class'}, inplace=True)
else:
    # If label column is named differently, update here
    pass

# Move 'class' column to the end
columns = list(df.columns)
if 'class' in columns:
    columns.remove('class')
    columns.append('class')
    df = df[columns]

# Separate features and labels
X = df.drop(columns=['class']).values
y = df['class'].values

# Apply UMAP to reduce dimensionality to 10 (adjust if needed)
n_umap_dims = 10
reducer = umap.UMAP(n_components=n_umap_dims)
X_umap = reducer.fit_transform(X)

# Create new DataFrame with UMAP features and class label
umap_columns = [f"umap_dim_{i+1}" for i in range(n_umap_dims)]
df_umap = pd.DataFrame(X_umap, columns=umap_columns)
df_umap['class'] = y

# Save the UMAP-processed dataset
output_filename = "covtype_umap10_with_class.csv"
output_path = os.path.join(SAVE_DATA_DIR, output_filename)
df_umap.to_csv(output_path, index=False)
