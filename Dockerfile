FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file ke container
COPY . .

# Jalankan training model (simpan model ke models/)
RUN python src/train_models.py

# Expose port FastAPI
EXPOSE 8000

# Jalankan aplikasi FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
