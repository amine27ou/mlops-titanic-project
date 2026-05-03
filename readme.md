# MLOps Titanic Survival Prediction

End-to-end Machine Learning Operations (MLOps) pipeline for predicting Titanic passenger survival using industry-standard tools and best practices.

## 🎯 Project Overview

This project implements a complete MLOps pipeline demonstrating:
- **Experiment Tracking** with MLflow
- **Data Versioning** with DVC
- **Pipeline Orchestration** with Apache Airflow
- **Model Deployment** with FastAPI + Docker
- **CI/CD** with GitHub Actions
- **Testing** with Pytest
- **Monitoring** with Grafana + Prometheus


---

## 📊 Dataset

**Source:** [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)  
**Size:** 891 passengers (training set)  
**Task:** Binary classification (Survived: 0/1)  
**Features:** Pclass, Sex, Age, SibSp, Parch, Fare

---

## 🏗️ Project Structure
mlops-titanic/
├── data/
│   ├── raw/                  # Original datasets (DVC tracked)
│   └── processed/            # Cleaned data
├── src/
│   ├── data/
│   │   ├── load_data.py      # Data loading utilities
│   │   └── preprocessing.py  # Data cleaning & feature engineering
│   ├── models/
│   │   ├── train.py          # Model training script
│   │   └── evaluate.py       # Model evaluation
│   └── eda.py                # EDA plot generation
├── tests/                    # Pytest unit tests
├── models/                   # Saved model artifacts
├── reports/
│   ├── figures/              # EDA plots, screenshots
│   └── baseline_results.csv  # Model comparison metrics
├── airflow/dags/             # Airflow DAGs
├── api/                      # FastAPI application
├── requirements.txt          # Python dependencies
└── README.md

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/mlops-titanic-project.git
cd mlops-titanic-project
```

### 2. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Data

1. Download from [Kaggle Titanic](https://www.kaggle.com/c/titanic/data)
2. Place `train.csv` and `test.csv` in `data/raw/`

### 4. Run EDA

```bash
python src/eda.py
```

Generates 5 plots in `reports/figures/`

### 5. Train Baseline Models

```bash
python src/models/train.py
```

Trains Logistic Regression and Random Forest, saves best model to `models/`

---

## 🛠️ Technology Stack

### Machine Learning
- **scikit-learn** - Model training
- **XGBoost** - Gradient boosting
- **pandas** - Data manipulation
- **matplotlib/seaborn** - Visualization

### MLOps Tools
- **MLflow** - Experiment tracking & model registry
- **DVC** - Data version control
- **Apache Airflow** - Pipeline orchestration
- **Great Expectations** - Data validation

### Deployment
- **FastAPI** - REST API
- **Docker** - Containerization
- **Kubernetes** - Orchestration (optional)

### CI/CD & Testing
- **GitHub Actions** - Continuous integration
- **Pytest** - Unit testing
- **Black/Flake8** - Code quality

### Monitoring
- **Prometheus** - Metrics collection
- **Grafana** - Dashboards
- **Evidently AI** - ML monitoring & drift detection

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t titanic-mlops .

# Run container
docker run -p 8000:8000 titanic-mlops
```

---

## 📚 Documentation

Full project documentation available in `reports/report.md`

- [Project Plan](mlops_project_plan.md)
- [Technical Report](reports/report.md) (In Progress)

---


---

## 📄 License

This project is part of an academic assignment.

---

## 🙏 Acknowledgments

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- MLOps community resources

---

