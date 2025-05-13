# %% Imports
import os
import pandas as pd

# %% Paths
# Get project root (assumes script is in a subfolder like 'utilities')
CURRENT_DIR = os.getcwd()
ROOT_PATH = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
DATA_DIR = os.path.join(ROOT_PATH, "data", "tabular")

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

# %% Main Script
if __name__ == "__main__":
    # Input: Set filename and class column name
    filename = "Seed_Data.csv"
    class_column = "target"

    # Load and rename
    df = load_and_rename_class_column(filename, class_column)

    # Save to new CSV
    output_filename = f"{os.path.splitext(filename)[0]}_class.csv"
    output_path = os.path.join(DATA_DIR, output_filename)
    df.to_csv(output_path, index=False)

    print(f"File saved to: {output_path}")
