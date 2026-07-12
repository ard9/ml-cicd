FROM python:3.11-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/

# استفاده برای train/promote — بدون نیاز به مدل Production
FROM base AS train

# استفاده برای سرویس‌دهی — مدل Production رو موقع build می‌گیره
FROM base AS api
RUN --mount=type=secret,id=mlflow_creds \
    export $(grep -v '^#' /run/secrets/mlflow_creds | xargs) && \
    python src/fetch_production_model.py
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]