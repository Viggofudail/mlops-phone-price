# Gunakan official Python slim image
FROM python:3.10-slim

# Set working directory di dalam container
WORKDIR /app

# Copy requirements.txt dulu supaya bisa install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file aplikasi ke container
COPY . .

# Expose port yang akan dipakai (FastAPI default 8000)
EXPOSE 8000

# Jalankan FastAPI dengan Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

