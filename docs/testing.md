# Testing Strategy

## Overview

This project uses pytest for unit testing with a Test-Driven Development (TDD) approach adapted for ML.

## Test Coverage

Current test coverage: **XX%** (run `pytest --cov=src`)

### Tested Modules

- `src/data/load_data.py` - Data loading (5 tests)
- `src/data/preprocessing.py` - Preprocessing pipeline (8 tests)

Total: **13 unit tests**

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # Mac
start htmlcov/index.html  # Windows
```

## Test Categories

### Unit Tests
- Test individual functions in isolation
- Mock external dependencies
- Fast execution (<1 second per test)

### Integration Tests (Future)
- Test end-to-end pipeline
- Test MLflow logging
- Test API endpoints

## Pre-commit Hooks

Code quality checks run automatically before each commit:

- **Black** - Code formatting
- **Flake8** - Linting
- **Trailing whitespace** - Cleanup
- **YAML validation** - Config files

To run manually:
```bash
pre-commit run --all-files
```

## ML-Specific Testing Challenges

### Challenges Addressed:

1. **Data Leakage Prevention**
   - Tests verify train/test split has no overlap
   - Tests check feature engineering doesn't use future data

2. **Reproducibility**
   - Tests verify random_state produces same results
   - Tests check preprocessing is deterministic

3. **Data Quality**
   - Tests verify no missing values after preprocessing
   - Tests check value ranges are valid

### Known Limitations:

- Model performance tests not included (non-deterministic)
- Integration tests pending (Day 5)
- API tests pending (Day 11)

## Common ML Bugs Prevented

✅ **Data Leakage** - Tests verify proper train/test splitting
✅ **Missing Values** - Tests ensure all nulls handled
✅ **Type Errors** - Type hints catch incorrect inputs
✅ **Shape Mismatches** - Tests verify expected dimensions
✅ **Encoding Errors** - Tests check categorical encoding

## CI/CD Integration

Tests run automatically in GitHub Actions on every push (Day 9).
