FROM python:3.10-slim

WORKDIR /app

# Copy and install dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model + app
COPY api/model/ /app/model/
COPY api/app.py /app/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
