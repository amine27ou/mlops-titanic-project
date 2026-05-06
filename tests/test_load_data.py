import pytest
import pandas as pd
from data.load_data import load_raw_data


def test_load_raw_data_exists():
    df = load_raw_data("data/raw/train.csv")

    assert isinstance(df, pd.DataFrame), "Should return DataFrame"
    assert len(df) > 0, "DataFrame should not be empty"


def test_load_raw_data_columns():
    """Test that loaded data has expected columns."""
    df = load_raw_data("data/raw/train.csv")

    expected_cols = [
        "PassengerId",
        "Survived",
        "Pclass",
        "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Ticket",
        "Fare",
        "Cabin",
        "Embarked",
    ]

    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_load_raw_data_shape():
    """Test that Titanic training data has correct shape."""
    df = load_raw_data("data/raw/train.csv")

    assert df.shape[0] == 891, "Titanic train.csv should have 891 rows"
    assert df.shape[1] == 12, "Should have 12 columns"


def test_load_raw_data_target_values():
    """Test that Survived column has correct values."""
    df = load_raw_data("data/raw/train.csv")

    assert "Survived" in df.columns, "Should have Survived column"
    assert set(df["Survived"].unique()).issubset({0, 1}), "Survived should be 0 or 1"


def test_load_raw_data_file_not_found():
    """Test that loading non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("data/raw/nonexistent.csv")
