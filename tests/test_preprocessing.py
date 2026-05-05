import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from data.preprocessing import preprocess_data, split_data

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_data():
    """Create sample Titanic data for testing."""
    data = {
        "PassengerId": [1, 2, 3, 4, 5],
        "Survived": [0, 1, 1, 1, 0],
        "Pclass": [3, 1, 3, 1, 3],
        "Sex": ["male", "female", "female", "female", "male"],
        "Age": [22.0, 38.0, 26.0, 35.0, np.nan],  # One missing value
        "SibSp": [1, 1, 0, 1, 0],
        "Parch": [0, 0, 0, 0, 0],
        "Fare": [7.25, 71.28, 7.92, 53.1, 8.05],
    }
    return pd.DataFrame(data)


def test_preprocess_data_shape(sample_data):
    """Test that preprocessing returns correct shapes."""
    X, y = preprocess_data(sample_data)

    assert X.shape[0] == 5, "X should have 5 rows"
    assert X.shape[1] == 6, "X should have 6 features"
    assert len(y) == 5, "y should have 5 values"


def test_preprocess_data_no_missing(sample_data):
    """Test that preprocessing handles missing values."""
    X, y = preprocess_data(sample_data)

    # No missing values should remain
    assert X.isnull().sum().sum() == 0, "Should have no missing values"


def test_preprocess_data_sex_encoding(sample_data):
    """Test that Sex is properly encoded as 0/1."""
    X, y = preprocess_data(sample_data)

    # Sex should be numeric
    assert X["Sex"].dtype in [np.int64, np.float64], "Sex should be numeric"

    # Should only contain 0 and 1
    assert set(X["Sex"].unique()).issubset({0, 1}), "Sex should be 0 or 1"


def test_preprocess_data_missing_columns():
    """Test that preprocessing raises error for missing columns."""
    bad_data = pd.DataFrame({"A": [1, 2, 3]})

    with pytest.raises(KeyError):
        preprocess_data(bad_data)


def test_split_data_shapes(sample_data):
    """Test that split_data returns correct shapes."""
    X, y = preprocess_data(sample_data)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.4, random_state=42)

    # 60% train, 40% test
    assert len(X_train) == 3, "Should have 3 training samples"
    assert len(X_test) == 2, "Should have 2 test samples"
    assert len(y_train) == 3, "Should have 3 training labels"
    assert len(y_test) == 2, "Should have 2 test labels"


def test_split_data_no_overlap(sample_data):
    """Test that train and test sets don't overlap."""
    X, y = preprocess_data(sample_data)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.4, random_state=42)

    # Indices should not overlap
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)

    assert (
        len(train_indices.intersection(test_indices)) == 0
    ), "Train and test should not overlap"


def test_split_data_invalid_test_size(sample_data):
    """Test that invalid test_size raises error."""
    X, y = preprocess_data(sample_data)

    with pytest.raises(ValueError):
        split_data(X, y, test_size=1.5)  # Invalid: > 1

    with pytest.raises(ValueError):
        split_data(X, y, test_size=0.0)  # Invalid: = 0


def test_preprocess_reproducibility(sample_data):
    """Test that preprocessing is deterministic."""
    X1, y1 = preprocess_data(sample_data.copy())
    X2, y2 = preprocess_data(sample_data.copy())

    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)
