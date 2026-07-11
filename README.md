# Iris Classifier — MLOps CI/CD Example

A minimal but realistic MLOps project: train a model, gate it on quality,
serve it via FastAPI, containerize it, and ship it through a GitHub Actions
CI/CD pipeline.

## Project structure

```
.
├── .github/workflows/ci-cd.yml   # the CI/CD pipeline itself
├── src/
│   ├── train.py                  # training + quality gate
│   ├── model.py                  # prediction wrapper
│   └── api.py                    # FastAPI serving layer
├── tests/
│   ├── test_train.py
│   └── test_api.py
├── models/                       # trained model + metadata (generated)
├── Dockerfile
└── requirements.txt
```

## Run it locally

```bash
pip install -r requirements.txt

# 1. Train the model (also runs the quality gate)
python src/train.py

# 2. Run tests
pytest tests/ -v

# 3. Lint
ruff check src/ tests/

# 4. Run the API
uvicorn src.api:app --reload
# then: curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
#   -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## Run it with Docker

```bash
docker build -t iris-api .
docker run -p 8000:8000 iris-api
```

## Run it with Docker Compose (recommended for local dev)

```bash
# Start the API with hot-reload (edits to src/ apply live)
docker compose up

# Retrain the model in a container, without touching the running API's build
docker compose run --rm train

# Stop everything
docker compose down
```

The `api` service mounts `src/` and `models/` as volumes, so editing code
or retraining doesn't require a rebuild — just a container restart (or
nothing at all, thanks to `--reload`).

## The CI/CD pipeline (`.github/workflows/ci-cd.yml`)

Runs on every push/PR to `main`, as 4 sequential jobs:

1. **lint-and-test** — ruff + fast unit tests. Fails fast on obviously bad code.
2. **train-and-validate** — retrains the model from scratch and enforces a
   **quality gate** (`MIN_ACCURACY = 0.90` in `src/train.py`). If the newly
   trained model doesn't clear the bar, the pipeline stops here — nothing
   bad ever reaches production. This is the "MLOps" part: the model itself
   is a build artifact with a test, just like your code.
3. **build-and-push** — only runs on `main`. Builds a Docker image using the
   *validated* model artifact from job 2, pushes it to GitHub Container
   Registry (`ghcr.io`), tagged both `latest` and with the commit SHA.
4. **deploy** — calls a Render deploy hook (via the `RENDER_DEPLOY_HOOK`
   secret) to trigger Render to pull the freshly pushed image and redeploy.
   If the secret isn't set, this step logs a message and exits cleanly
   instead of failing the pipeline.

## Deploying to Render (free tier)

1. Push to GitHub once so the pipeline runs and publishes an image to GHCR.
2. On GitHub, make the package public: your profile → **Packages** → the
   package → **Package settings** → **Change visibility → Public**.
   (Public avoids having to configure a registry credential on Render.)
3. On [render.com](https://render.com), sign up with GitHub, then
   **New → Web Service → Existing Image**, and point it at
   `ghcr.io/<you>/<repo>:latest`, port `8000`, plan **Free**.
4. Once created, go to the service's **Settings → Deploy Hook** and copy
   the URL.
5. In your GitHub repo: **Settings → Secrets and variables → Actions →
   New repository secret**, name it `RENDER_DEPLOY_HOOK`, paste the URL.
6. Push again (or re-run the workflow) — the `deploy` job will now call
   the hook and Render will pull the new image automatically.

Note: Render's free web services spin down after inactivity and take
~30-60s to wake up on the next request — normal for the free tier, not a bug.

## To use this on your own GitHub repo

1. `git init && git add . && git commit -m "initial commit"`
2. Create a repo on GitHub and push
3. Nothing else to configure — `GITHUB_TOKEN` (used to push to GHCR) is
   provided automatically by GitHub Actions
4. Go to the **Actions** tab and watch the pipeline run
5. Your image will show up under your GitHub profile's **Packages** tab

## Ideas to extend this (good for interview talking points)

- Swap the toy Iris dataset for a real one, or add MLflow for experiment tracking
- Add a `staging` environment with manual approval before `production` deploy
- Add integration tests that hit the running Docker container, not just the app object
- Add drift detection: a scheduled job that re-evaluates the model against fresh data
- Use `dependabot.yml` to keep dependencies patched automatically
