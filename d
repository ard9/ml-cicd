[1mdiff --git a/.dockerignore b/.dockerignore[m
[1mindex 671bb25..6cb365c 100644[m
[1m--- a/.dockerignore[m
[1m+++ b/.dockerignore[m
[36m@@ -6,3 +6,8 @@[m [m__pycache__[m
 tests/[m
 README.md[m
 .venv[m
[32m+[m[32m.env[m
[32m+[m[32mmlflow_creds.txt[m
[32m+[m[32mmlruns/[m
[32m+[m[32mmlflow-db/[m
[32m+[m[32mmodels/[m
\ No newline at end of file[m
[1mdiff --git a/Dockerfile b/Dockerfile[m
[1mindex f54b236..b35c066 100644[m
[1m--- a/Dockerfile[m
[1m+++ b/Dockerfile[m
[36m@@ -1,18 +1,16 @@[m
[31m-FROM python:3.11-slim[m
[31m-[m
[32m+[m[32mFROM ml-cicd-project-api:v2 AS base[m
 WORKDIR /app[m
[31m-[m
[31m-# Install dependencies first (better layer caching)[m
 COPY requirements.txt .[m
 RUN pip install --no-cache-dir -r requirements.txt[m
[31m-[m
[31m-# Copy application code[m
 COPY src/ src/[m
 [m
[31m-# Model is trained + validated in CI before the image is built,[m
[31m-# then copied in here. This keeps the image build fast and reproducible.[m
[31m-COPY models/ models/[m
[32m+[m[32m# استفاده برای train/promote — بدون نیاز به مدل Production[m
[32m+[m[32mFROM base AS train[m
 [m
[32m+[m[32m# استفاده برای سرویس‌دهی — مدل Production رو موقع build می‌گیره[m
[32m+[m[32mFROM base AS api[m
[32m+[m[32mRUN --mount=type=secret,id=mlflow_creds \[m
[32m+[m[32m    export $(grep -v '^#' /run/secrets/mlflow_creds | xargs) && \[m
[32m+[m[32m    python src/fetch_production_model.py[m
 EXPOSE 8000[m
[31m-[m
[31m-CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"][m
[32m+[m[32mCMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"][m
\ No newline at end of file[m
[1mdiff --git a/docker-compose.yml b/docker-compose.yml[m
[1mindex b68153b..e31e630 100644[m
[1m--- a/docker-compose.yml[m
[1m+++ b/docker-compose.yml[m
[36m@@ -1,11 +1,13 @@[m
 services:[m
   api:[m
[31m-    build: .[m
[32m+[m[32m    build:[m
[32m+[m[32m      context: .[m
[32m+[m[32m      secrets:[m
[32m+[m[32m        - mlflow_creds[m
[32m+[m[32m    image: ml-cicd-project-api:v2[m
     ports:[m
       - "8000:8000"[m
     volumes:[m
[31m-      # Mount source + model so you can edit code and see changes without rebuilding.[m
[31m-      # Remove these two lines for a "production-like" run using only what's baked into the image.[m
       - ./src:/app/src[m
       - ./models:/app/models[m
     environment:[m
[36m@@ -22,8 +24,44 @@[m [mservices:[m
   # Writes to ./models, which the api service mounts above, so a fresh[m
   # training run is picked up by the running API without a rebuild.[m
   train:[m
[31m-    build: .[m
[32m+[m[32m    build:[m
[32m+[m[32m      context: .[m
[32m+[m[32m      target: train[m
[32m+[m[32m    image: ml-cicd-project-api:train[m
[32m+[m[32m    env_file: .env[m
     volumes:[m
       - ./models:/app/models[m
[32m+[m[32m      - ./mlruns:/mlruns[m
     command: python src/train.py[m
     profiles: ["tools"][m
[32m+[m[32m    depends_on: [mlflow][m
[32m+[m
[32m+[m[32m  mlflow:[m
[32m+[m[32m    image: ghcr.io/mlflow/mlflow:v2.15.1[m
[32m+[m[32m    container_name: mlflow[m
[32m+[m[32m    ports:[m
[32m+[m[32m      - "5000:5000"[m
[32m+[m[32m    volumes:[m
[32m+[m[32m      - ./mlruns:/mlruns[m
[32m+[m[32m      - ./mlflow-db:/mlflow-db[m
[32m+[m[32m    command:[m
[32m+[m[32m      - mlflow[m
[32m+[m[32m      - server[m
[32m+[m[32m      - --host[m
[32m+[m[32m      - 0.0.0.0[m
[32m+[m[32m      - --port[m
[32m+[m[32m      - "5000"[m
[32m+[m[32m      - --backend-store-uri[m
[32m+[m[32m      - sqlite:////mlflow-db/mlflow.db[m
[32m+[m[32m      - --artifacts-destination[m
[32m+[m[32m      - /mlruns[m
[32m+[m[32m      - --serve-artifacts[m
[32m+[m[32m      - --gunicorn-opts[m
[32m+[m[32m      - "--access-logfile - --error-logfile -"[m
[32m+[m[32msecrets:[m
[32m+[m[32m  mlflow_creds:[m
[32m+[m[32m    file: ./mlflow_creds.txt[m
[32m+[m
[32m+[m[32mnetworks:[m
[32m+[m[32m    ml-network:[m
[32m+[m[32m      driver: bridge[m
\ No newline at end of file[m
