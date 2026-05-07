# Titanic MLOps Project

## Project Overview
This repository contains an end-to-end MLOps pipeline for Titanic survival prediction.
It includes data loading and preprocessing, model training, experiment tracking, pipeline orchestration, containerized model serving, and monitoring.

Key features:
- FastAPI model serving in `api/app.py`
- Dockerized deployment for the API in `api/Dockerfile`
- Airflow orchestration via `docker-compose-airflow.yml`
- MLflow experiment tracking and artifacts
- Data versioning with DVC
- Unit tests in `tests/`
- Prometheus-style metrics for API monitoring

## Repository Structure

- `airflow/`
  - `dags/` - Airflow DAG definitions
  - `logs/` - Airflow logs
- `api/`
  - `app.py` - FastAPI application
  - `requirements.txt` - API dependencies
  - `Dockerfile` - API container definition
  - `model/` - trained model artifacts
- `data/`
  - `raw/` - raw datasets
  - `processed/` - cleaned dataset outputs
- `docs/` - documentation files and setup guides
- `models/` - saved model artifacts and exports
- `src/`
  - `data/` - data loading and preprocessing modules
  - `models/` - training and evaluation logic
  - `utils/` - helper utilities
- `tests/` - Pytest unit tests
- `docker-compose-airflow.yml` - Airflow service definition
- `Dockerfile` - root Dockerfile for Airflow / base image usage
- `requirements.txt` - repository-level Python dependencies
- `readme.md` - high-level project overview

## Current Context

This project is designed to be used in two main ways:
1. **Run the FastAPI model serving application** for inference and monitoring.
2. **Run the Airflow orchestration environment** for pipeline execution.

The API and Airflow setups are separate but share the same codebase and dataset.

## Prerequisites

- Python 3.9+ or a compatible virtual environment
- Docker and Docker Compose
- Git
- Optional: DVC and MLflow if you want full experiment and data tracking

## Setup and Local Development

### 1. Create and activate a Python environment

Windows PowerShell:
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare data

The repository expects the Titanic dataset files to be available under `data/raw/`.
Place the following files in `data/raw/`:
- `train.csv`
- `test.csv`

If DVC is configured, you may also pull data with:
```bash
dvc pull
```

### 3. Train or evaluate the model

Model training and evaluation scripts are available in `src/models/`.

Example commands:
```bash
python src/models/train.py
```

If the repo includes MLflow tracking, use the existing MLflow commands or logs to inspect experiments.

## Run the FastAPI Application Locally

This will start the API without Docker, using the local source tree.

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Then open:
- `http://localhost:8000/` for root health info
- `http://localhost:8000/health` for health status
- `http://localhost:8000/metrics` for Prometheus metrics
- `http://localhost:8000/docs` for interactive Swagger UI

## Run the FastAPI API in Docker

The API Dockerfile is located in `api/Dockerfile` and must be built from the project root.

```powershell
cd C:\Users\starkiller\Documents\py\mlops-titanic
docker build -t titanic-api:latest -f api/Dockerfile .
docker run -d -p 8000:8000 --name titanic-api titanic-api:latest
```

Then test:
```powershell
curl http://localhost:8000/metrics
```

### Important note
The API Dockerfile uses `ENV MODEL_PATH=/app/model/model.pkl` and copies the model into `/app/model/model.pkl`.
Make sure that model file exists in `api/model/model.pkl` before building.

## Run Apache Airflow via Docker Compose

The `docker-compose-airflow.yml` file defines a single Airflow service that mounts project folders into the container.

Start Airflow:

```bash
docker compose -f docker-compose-airflow.yml up -d
```

Then access Airflow UI at:
- `http://localhost:8080`

Default credentials from the compose file:
- username: `admin`
- password: `admin`

To stop Airflow:

```bash
docker compose -f docker-compose-airflow.yml down
```

## Testing

Run unit tests with Pytest:

```bash
pytest tests/ -v
```

If coverage is configured:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Monitoring and Metrics

The API exposes Prometheus metrics at `/metrics`.
It also includes application-level instrumentation for prediction counts and duration.

Example endpoint usage:
- `GET /metrics`
- `GET /health`
- `POST /predict` with JSON payload matching `PassengerData`

## Notes and Troubleshooting

- If `curl http://localhost:8000/metrics` returns `404`, confirm the correct container is running and that the API image was built from the repo root using the correct `api/Dockerfile`.
- Do not build the API image with `docker build -t titanic-api:latest ./api`; that uses the wrong build context for this Dockerfile.
- If the container fails to start with `No such file or directory: './api/model/model.pkl'`, make sure the active code and `MODEL_PATH` are set correctly for Docker. The API expects `/app/model/model.pkl` inside the container.

## Helpful Files

- `api/app.py` — API code and prediction logic
- `api/requirements.txt` — API dependencies
- `api/Dockerfile` — API deployment image
- `docker-compose-airflow.yml` — Airflow orchestration service
- `src/` — data and model logic
- `tests/` — automated tests
- `data/` — raw and processed dataset storage

## Summary
This project combines MLOps best practices with a production-style deployment stack:
- build/train models with Python and scikit-learn/XGBoost
- track experiments with MLflow
- version data with DVC
- orchestrate with Airflow
- serve models with FastAPI and Docker
- monitor metrics with Prometheus-compatible endpoints

Use this `project.md` as a single reference for how the project is organized and how to run the major components.
