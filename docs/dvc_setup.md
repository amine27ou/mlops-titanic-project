# DVC (Data Version Control) Setup

## Overview

DVC is used to version control datasets and create reproducible ML pipelines.

## What is Tracked

### Raw Data
- `data/raw/train.csv` - Original Titanic training data (891 rows)
- `data/raw/test.csv` - Original Titanic test data (418 rows)

### Processed Data
- `data/processed/full_processed.csv` - Cleaned full dataset
- `data/processed/train_processed.csv` - Training split (80%)
- `data/processed/test_processed.csv` - Test split (20%)

## DVC Files in Git

DVC stores metadata in `.dvc` files that Git tracks:
- `data/raw/train.csv.dvc`
- `data/raw/test.csv.dvc`
- `data/processed/*.csv.dvc`

The actual data files are in `.gitignore` and managed by DVC.

## Reproducing the Dataset

To get the exact dataset used in this project:

```bash
# Pull data from DVC storage
dvc pull
```

## Pipeline Stages

The `dvc.yaml` file defines our ML pipeline:

1. **preprocess**: Cleans raw data → Creates processed datasets
2. **train**: Trains models using processed data → Generates metrics

## Running the Pipeline

```bash
# Run entire pipeline
dvc repro

# Check pipeline status
dvc status

# Visualize pipeline
dvc dag
```

## Benefits

- ✅ Dataset versioning (like Git for data)
- ✅ Reproducible experiments
- ✅ Efficient storage (only metadata in Git)
- ✅ Easy collaboration (share exact datasets)
- ✅ Pipeline automation
