FROM python:3.9-slim

WORKDIR /app

# Copy and install dependencies first (better layer caching)
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY api/app.py .
COPY api/model/ ./model/

# Copy preprocessing and config (very important for predictions to work)
COPY src/data/preprocessing.py ./src/data/
COPY src/utils/config.py ./src/utils/

# Create directories if needed
RUN mkdir -p src/data src/utils model

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
