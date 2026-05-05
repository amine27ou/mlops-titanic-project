"""
Data loading utilities for Titanic dataset.
"""

import pandas as pd


def load_raw_data(data_path: str = "data/raw/train.csv") -> pd.DataFrame:
    """
    Load raw Titanic training data.

    Args:
        data_path: Path to CSV file

    Returns:
        DataFrame with raw data
    """
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    return df


def get_data_overview(df: pd.DataFrame) -> None:
    """
    Print overview of dataset.

    Args:
        df: Input DataFrame
    """
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nTarget distribution:\n{df['Survived'].value_counts()}")
    print(f"\nSurvival rate: {df['Survived'].mean():.2%}")


if __name__ == "__main__":
    # Test the functions
    df = load_raw_data()
    get_data_overview(df)
