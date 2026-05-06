FROM python:3.9-slim

WORKDIR /app

# Copy and install dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application
COPY api/app.py .

# Copy the model
COPY api/model/ ./model/

# Copy source code needed for preprocessing (only if files exist)
COPY src/data/preprocessing.py ./src/data/ 2>/dev/null || true

# Only copy config if it exists
COPY src/utils/config.py ./src/utils/ 2>/dev/null || true

# Create directories structure
RUN mkdir -p src/data src/utils model

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
