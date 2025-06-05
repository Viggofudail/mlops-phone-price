FROM python:3.10-slim

RUN pip install mlflow==2.12.1

EXPOSE 5000

CMD ["mlflow", "server", "--backend-store-uri", "/mlflow/mlruns", "--default-artifact-root", "/mlflow/mlruns", "--host", "0.0.0.0", "--port", "5000"]
