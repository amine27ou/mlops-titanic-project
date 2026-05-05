"""
Data preprocessing for Titanic dataset.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess Titanic data for modeling.

    Steps:
    1. Select relevant features
    2. Handle missing values
    3. Encode categorical variables

    Args:
        df: Raw DataFrame with Titanic data

    Returns:
        Tuple of (X, y) where:
            - X: Feature matrix (DataFrame)
            - y: Target vector (Series)

    Raises:
        KeyError: If required columns are missing
        ValueError: If data contains unexpected values

    Example:
        >>> df = pd.read_csv('train.csv')
        >>> X, y = preprocess_data(df)
        >>> print(X.shape)
        (891, 6)
    """
    # Validate required columns exist
    required_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    # Select features
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    X = df[features].copy()
    y = df["Survived"].copy()

    # Handle missing values
    X["Age"].fillna(X["Age"].median(), inplace=True)
    X["Fare"].fillna(X["Fare"].median(), inplace=True)

    # Encode categorical variables
    X["Sex"] = X["Sex"].map({"male": 0, "female": 1})

    # Validate no missing values remain
    if X.isnull().sum().sum() > 0:
        raise ValueError("Preprocessing failed: missing values remain")

    print(f"Preprocessed data shape: {X.shape}")
    print(f"Missing values: {X.isnull().sum().sum()}")

    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets with stratification.

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for test set (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)

    Raises:
        ValueError: If test_size not between 0 and 1

    Example:
        >>> X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        >>> print(len(X_train), len(X_test))
        712 179
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))
    from data.load_data import load_raw_data

    # Test preprocessing
    df = load_raw_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n✅ Preprocessing successful!")
