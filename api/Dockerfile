FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (better caching)
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/app.py .

# Copy model
COPY api/model/model.pkl ./model/model.pkl

# Copy src if you have additional modules (preprocessing, utils, etc.)
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p ./model ./src/data ./src/utils

# Set proper environment variable for model path
ENV MODEL_PATH=/app/model/model.pkl

EXPOSE 8000

# Use modern uvicorn command with proper module
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
