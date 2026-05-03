"""
Data preprocessing for Titanic dataset.
"""

import pandas as pd
import numpy as np
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
        df: Raw DataFrame
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    # Select features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = df[features].copy()
    y = df['Survived'].copy()
    
    # Handle missing values
    X['Age'].fillna(X['Age'].median(), inplace=True)
    X['Fare'].fillna(X['Fare'].median(), inplace=True)
    
    # Encode categorical variables
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
    
    print(f"Preprocessed data shape: {X.shape}")
    print(f"Missing values: {X.isnull().sum().sum()}")
    
    return X, y


def split_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from load_data import load_raw_data
    
    # Test preprocessing
    df = load_raw_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("\n✅ Preprocessing successful!")