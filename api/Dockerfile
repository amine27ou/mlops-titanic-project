FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI app
COPY api/app.py .

# Copy trained model
COPY api/model/ ./model/

# Copy source code (preprocessing + utils)
COPY src/ ./src/

RUN mkdir -p ./model ./src/data ./src/utils

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
