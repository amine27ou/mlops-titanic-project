from pathlib import Path
from src.data.load_data import load_raw_data

# Get the test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_raw_data_exists():
    """Test that data loads without errors"""
    df = load_raw_data(str(FIXTURES_DIR / "train_sample.csv"))
    assert df is not None


def test_load_raw_data_columns():
    """Test that required columns exist"""
    df = load_raw_data(str(FIXTURES_DIR / "train_sample.csv"))
    required_columns = [
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
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"


def test_load_raw_data_shape():
    """Test data has expected shape"""
    df = load_raw_data(str(FIXTURES_DIR / "train_sample.csv"))
    assert df.shape[0] > 0, "DataFrame is empty"
    assert df.shape[1] == 12, f"Expected 12 columns, got {df.shape[1]}"


def test_load_raw_data_target_values():
    """Test target variable has expected values"""
    df = load_raw_data(str(FIXTURES_DIR / "train_sample.csv"))
    assert set(df["Survived"].dropna().unique()).issubset({0, 1})
