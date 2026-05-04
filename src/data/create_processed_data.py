"""
Create processed/cleaned datasets and track with DVC.
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.load_data import load_raw_data
from data.preprocessing import preprocess_data, split_data


def create_processed_datasets():
    """
    Load raw data, preprocess, split, and save to data/processed/
    """
    print("="*60)
    print("CREATING PROCESSED DATASETS")
    print("="*60)
    
    # Create processed directory
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True, parents=True)
    
    # Load raw data
    print("\nLoading raw data...")
    df = load_raw_data("data/raw/train.csv")
    
    # Preprocess
    print("\nPreprocessing data...")
    X, y = preprocess_data(df)
    
    # Combine for saving
    processed_df = X.copy()
    processed_df['Survived'] = y
    
    # Split into train/test
    print("\nSplitting into train/test...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    
    # Combine splits
    train_df = X_train.copy()
    train_df['Survived'] = y_train
    
    test_df = X_test.copy()
    test_df['Survived'] = y_test
    
    # Save datasets
    print("\nSaving processed datasets...")
    
    # Full processed dataset
    full_path = processed_dir / "full_processed.csv"
    processed_df.to_csv(full_path, index=False)
    print(f"✅ Saved: {full_path}")
    
    # Train set
    train_path = processed_dir / "train_processed.csv"
    train_df.to_csv(train_path, index=False)
    print(f"✅ Saved: {train_path}")
    
    # Test set
    test_path = processed_dir / "test_processed.csv"
    test_df.to_csv(test_path, index=False)
    print(f"✅ Saved: {test_path}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Full dataset: {processed_df.shape}")
    print(f"Train set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")
    print(f"\nSurvival distribution (train):")
    print(train_df['Survived'].value_counts(normalize=True))
    
    print(f"\n{'='*60}")
    print("✅ PROCESSED DATASETS CREATED")
    print(f"{'='*60}")


if __name__ == "__main__":
    create_processed_datasets()