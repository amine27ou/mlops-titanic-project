FROM python:3.9-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Main app files
COPY api/app.py .
COPY api/model/ ./model/

# Source code for prediction logic
COPY src/ ./src/

# Ensure directories exist
RUN mkdir -p model

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
