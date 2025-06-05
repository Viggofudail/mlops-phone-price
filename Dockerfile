FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy aplikasi FastAPI + kode MLflow jika ada
COPY . .

# Expose port FastAPI dan MLflow server (default 5000)
EXPOSE 8000 5000

# Jalankan MLflow server di background, lalu jalankan FastAPI di foreground
CMD mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 & \
    uvicorn app.main:app --host 0.0.0.0 --port 8000
