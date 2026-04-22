# Proactive AI Unified Platform

Proactive AI Unified Platform is a recommendation-system workbench that combines:

- automated dataset ingestion and column detection
- benchmark and hyperparameter tuning across multiple recommendation algorithms
- model registry, promotion, and serving APIs
- a React frontend for training and recommendation workflows
- a smart database-to-dataset builder that can turn connected data sources into training-ready CSV or JSON datasets

The repository supports both CLI-style offline training and a FastAPI application that serves training, model management, recommendation, and dataset-building endpoints.

## What This Project Does

At a high level, the project helps you move from raw interaction data to a serving-ready recommender:

1. Load a dataset from file, SQL, or NoSQL.
2. Detect user, item, rating, and timestamp columns automatically.
3. Profile feedback type and resolve the training mode (`explicit`, `implicit`, or `auto`).
4. Clean and split the data.
5. Benchmark multiple recommendation algorithms.
6. Tune the top-ranked candidates with Optuna.
7. Save versioned models to the local registry.
8. Promote the best model for serving.
9. Serve recommendations through authenticated APIs or the frontend.

The same app also includes a Smart DB CSV Builder flow:

1. Register database connections.
2. Inspect schemas.
3. Build a recommendation dataset from selected sources in background jobs.
4. Load the generated CSV into the training workflow.

## Main Features

- FastAPI backend with training, jobs, model registry, recommendation, and explainability endpoints
- React frontend for source selection, DB connection management, dataset build, training, and recommendation
- CLI training mode for one-shot model benchmarking from local files
- Background job execution for file training and dataset-building workflows
- Algorithm registry with explicit, implicit, hybrid, domain-specific, and content-aware recommenders
- Model registry with versioning and per-algorithm promotion
- Recommendation strategies:
  - `best_promoted_model`
  - `single_model`
  - `ensemble_weighted`
- Plain-language model-selection summaries in the UI
- Docker and Docker Compose support
- Test coverage for recommendation strategy logic and UI payload helpers

## Architecture Overview

The application is organized around a few core subsystems:

- `main.py`
  Starts either the FastAPI server or the CLI training flow.

- `app_factory.py`
  Builds the unified FastAPI app, mounts static assets, serves the built frontend, and wires both the recommendation API and Smart DB CSV Builder routers.

- `api/`
  Main recommendation and model-serving API.

- `pipeline/`
  End-to-end training pipeline, including benchmark orchestration and model selection policy generation.

- `recommendation/`
  Serving-time recommendation orchestration and strategy selection.

- `models/`
  Model registry and model loading.

- `data_processing/`
  Dataset analysis, feedback detection, cleaning, and interaction matrix creation.

- `smart_db_csv_builder/`
  Secondary module for schema inspection and recommendation dataset generation from live databases.

- `frontend/`
  Vite + React app used for the main interactive workflow.

## Repository Layout

```text
.
├── algorithms/               # Algorithm registry and implementations
├── api/                      # Main FastAPI routes and auth
├── benchmark/                # Benchmark engine
├── config/                   # Settings and DB config helpers
├── data_processing/          # Dataset analysis, cleaning, interaction matrix
├── frontend/                 # React + Vite frontend
├── ingestion/                # File, SQL, and NoSQL ingestion helpers
├── insights/                 # Explainability logic
├── models/                   # Model registry and loader
├── optimization/             # Optuna tuning
├── pipeline/                 # End-to-end training pipeline
├── recommendation/           # Recommendation strategy and serving logic
├── smart_db_csv_builder/     # DB schema + dataset builder module
├── static/                   # Swagger UI assets
├── tests/                    # Python and JS tests
├── ui/                       # Legacy dashboard assets and JS helpers
├── app_factory.py            # Unified FastAPI application factory
├── main.py                   # CLI and server entrypoint
├── requirements.txt          # Python dependencies
├── docker_compose.yml        # API + CLI compose setup
└── Dockerfile                # Multi-stage image build
```

## Supported Algorithms

The central algorithm registry lives in `algorithms/__init__.py`. At the time of writing, it includes:

- `SVD`
- `SVD++`
- `NMF`
- `ALS`
- `BPR`
- `User-KNN`
- `Item-KNN`
- `EASE`
- `LightFM`
- `Autoencoder-CF`
- `Content-Based TF-IDF`
- `Factorization Machines`
- `Ecommerce-Popularity`
- `Ecommerce-Purchase-ALS`
- `Movie-Item-KNN`
- `Temporal-SVD`

Each algorithm carries metadata such as feedback compatibility, scalability, robustness, sparsity fit, interpretability, and production readiness. The benchmark and recommendation layers use that metadata when ranking and explaining model choices.

## Data Expectations

The training flow tries to auto-detect these roles:

- `userID`
- `itemID`
- `rating`
- `timestamp`

Column detection is fuzzy and includes fallback behavior. The dataset analyzer also validates likely ID and rating columns and emits warnings for suspicious mappings.

The platform supports:

- explicit-feedback datasets
- implicit-feedback datasets
- hybrid datasets

The resolved mode is determined automatically unless you override it with `explicit` or `implicit`.

## Backend APIs

### Public endpoints

- `GET /health`
- `GET /session/status`
- `GET /algorithms`
- `POST /auth/login`
- `POST /train/file`
- `POST /train/sql`
- `GET /jobs/{job_id}`

### Protected endpoints

- `GET /auth/me`
- `POST /recommend`
- `POST /recommend/batch`
- `GET /recommend/options`
- `GET /similar/{item_id}`
- `GET /explain/{user_id}/{item_id}`
- `GET /models`
- `POST /models/{model_id}/promote`

### Smart DB CSV Builder endpoints

- `POST /smart-db-csv/api/connections/test`
- `POST /smart-db-csv/api/connections`
- `GET /smart-db-csv/api/connections`
- `DELETE /smart-db-csv/api/connections/{conn_id}`
- `GET /smart-db-csv/api/schema/{conn_id}`
- `POST /smart-db-csv/api/build`
- `GET /smart-db-csv/api/jobs/{job_id}`
- `GET /smart-db-csv/api/jobs/{job_id}/download`

## Authentication

The backend uses a simple credential-based login that issues bearer tokens.

Default credentials from `config/settings.py`:

- username: `admin`
- password: `admin123`

These should be overridden in non-local environments.

## Local Development

### 1. Python environment

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the backend

```powershell
venv\Scripts\python.exe main.py --serve
```

Or simply:

```powershell
venv\Scripts\python.exe main.py
```

Default API URL:

- `http://127.0.0.1:8000`

Swagger UI:

- `http://127.0.0.1:8000/docs`

### 3. Start the frontend

```powershell
cd frontend
npm install
npm run dev
```

Default frontend URL:

- `http://127.0.0.1:5173`

The Vite dev server proxies backend requests to `http://127.0.0.1:8000` by default. You can override the proxy target with:

```powershell
$env:VITE_API_PROXY_TARGET="http://127.0.0.1:8000"
```

### 4. Production-style frontend build

```powershell
cd frontend
npm run build
```

When the build output exists in `frontend/dist`, the FastAPI app serves it from `/`.

## CLI Usage

You can run training directly from the command line without using the frontend.

Example:

```powershell
venv\Scripts\python.exe main.py --file .\data\dataset.csv --topk 10 --top-models 10 --trials -1 --mode auto
```

Important CLI flags:

- `--file`
- `--format`
- `--topk`
- `--top-models`
- `--trials`
- `--mode`
- `--interactive`
- `--force-all`
- `--no-tune`
- `--serve`

## Training Flow

The training pipeline is implemented in `pipeline/training_pipeline.py`.

It runs these stages:

1. Column detection
2. Feedback profiling
3. Data cleaning
4. Interaction matrix creation
5. Benchmarking
6. Hyperparameter tuning

After tuning, the pipeline:

- ranks tuned candidates
- saves model artifacts into `model_store/`
- auto-promotes the top-ranked model when enabled
- produces a model-selection policy for the frontend and recommendation layer

## Recommendation Flow

Recommendation requests are handled through `recommendation/strategy_service.py`.

The system supports:

- best promoted model
- specific saved model
- weighted ensemble across shortlisted models

The service returns:

- the models used
- normalized weights if applicable
- contribution breakdowns
- final ranked items
- warnings when fallbacks occur

## Smart DB CSV Builder

The `smart_db_csv_builder/` module lets you create a recommendation-ready dataset from connected databases.

Key concepts:

- connection registration and testing
- schema inspection
- background build jobs
- export to CSV or JSON
- build modes:
  - `query`
  - `manual`
  - `llm`

Supported builder-side DB types in the schema models:

- MySQL
- PostgreSQL
- SQL Server
- SQLite
- MongoDB
- Redis

Builder requests support recommendation system targets such as:

- collaborative
- content-based
- hybrid
- sequential

## Important Configuration

Main runtime settings live in `config/settings.py`.

Notable environment variables:

- `API_HOST`
- `API_PORT`
- `TOP_K`
- `TOP_N_MODELS`
- `DEFAULT_ALGORITHM_MODE`
- `OPTUNA_TIMEOUT_S`
- `MODEL_STORE_DIR`
- `OPTUNA_SQLITE_PATH`
- `AUTH_USERNAME`
- `AUTH_PASSWORD`
- `AUTH_SECRET_KEY`
- `AUTH_TOKEN_TTL_MINUTES`
- `LOG_LEVEL`
- `LOG_FILE`

Important constraints:

- `TOP_K_ALLOWED = (5, 10)`
- `TOP_MODEL_ALLOWED = (5, 10)`

## Model Storage

Saved models are written to `model_store/`.

The registry:

- versions models per algorithm
- stores metadata and metrics in `model_store/registry.json`
- tracks promotion status
- resolves model files across relative and absolute paths

## Docker

The repository includes:

- `Dockerfile`
- `docker_compose.yml`

Compose services:

- `api`
  Runs the FastAPI app on port `8000`
- `cli`
  Runs one-shot CSV benchmarking as a profile-based service

Example:

```powershell
docker compose up api
```

To run the CLI profile:

```powershell
docker compose --profile cli up cli
```

## Testing

Current test files:

- `tests/test_recommendation_strategy_service.py`
- `tests/ui_recommendation_ui_logic.test.js`

Suggested local commands:

```powershell
venv\Scripts\python.exe -m unittest tests.test_recommendation_strategy_service
```

```powershell
node tests\ui_recommendation_ui_logic.test.js
```

## Development Notes

- The unified FastAPI app serves both the recommendation API and the Smart DB CSV Builder API.
- The frontend supports automatic flow transitions after dataset build and training.
- Recommendation UI logic now reflects the selected shortlist size instead of hardcoding `5`.
- The project currently uses simple local auth suitable for development, not a production IAM setup.
- Some directories such as `model_store/`, `results/`, and `logs/` are runtime artifact directories and will change as models are trained.

## Typical Workflow

### File-based workflow

1. Start backend.
2. Start frontend.
3. Open the UI.
4. Choose CSV mode.
5. Upload a dataset.
6. Train the top 5 or top 10 models.
7. Review the shortlisted models and model-selection explanation.
8. Log in and request recommendations.

### Database-based workflow

1. Start backend.
2. Start frontend.
3. Choose Database mode.
4. Add and test one or more DB connections.
5. Inspect schemas.
6. Build a dataset from the selected sources.
7. Auto-load the generated CSV into training.
8. Train models.
9. Review the shortlist and request recommendations.

## Current Status

This repository is an active working project with:

- backend logic
- frontend UI
- dataset builder
- saved model artifacts
- local result files

If you are using it as a base for production work, review:

- secret management
- authentication hardening
- file path handling for model artifacts
- dependency pinning
- CI test automation

