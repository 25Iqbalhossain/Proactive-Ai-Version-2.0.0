# Project Overview And Bug Report

## Overview

This repository is a unified recommendation-system platform with four main parts:

- `api/`, `app_factory.py`, `main.py`: FastAPI application, auth, training endpoints, recommendation endpoints, and job orchestration.
- `pipeline/`, `benchmark/`, `optimization/`, `models/`: end-to-end training flow, algorithm benchmarking, Optuna tuning, persistence, and serving selection.
- `recommendation/`, `insights/`: serving-time recommendation logic, strategy selection, and explanation support.
- `smart_db_csv_builder/` and `frontend/`: database-to-dataset builder plus the React UI used to run training and recommendation workflows.

At a high level, the intended flow is:

1. Build or upload a dataset.
2. Detect columns and resolve feedback mode.
3. Benchmark and tune candidate recommenders.
4. Save and promote models in `model_store/`.
5. Serve recommendations through the API or frontend.

## Review Scope

I reviewed the repository structure, README, main backend entrypoints, training pipeline, benchmark engine, recommendation strategy service, model registry/loader, auth, and the main React app. I also ran the Python unit tests that exist in the repo.

## Findings

### 1. CLI mode is effectively broken

- Severity: High
- Files: `main.py:25-40`, `main.py:77-78`

`main.py` defines a real CLI path in `main()`, but the module entrypoint never calls it. Running the file directly executes `_start_server()` unconditionally:

- `main.py:39-40` routes CLI execution to training when `--file` is provided.
- `main.py:77-78` bypasses that logic and starts the API server directly.

Impact:

- The README's CLI usage is misleading in practice.
- `python main.py --file ...` will start Uvicorn instead of running training.
- `--help`, `--topk`, `--mode`, and other CLI flags are effectively dead when the file is executed normally.

### 2. `hybrid` mode is documented and partially implemented, but validation rejects it

- Severity: High
- Files: `config/settings.py:58`, `api/routes.py:212-213`, `api/routes.py:528-529`, `pipeline/training_pipeline.py:58-60`, `benchmark/benchmark_engine.py:24`, `benchmark/benchmark_engine.py:249-250`, `README.md:146`, `README.md:343`

The system documentation and benchmark engine both treat `hybrid` as a supported mode, but global validation does not:

- `config/settings.py:58` sets `ALGORITHM_MODES = ("explicit", "implicit", "auto")`
- API and training config validation use that tuple, so `hybrid` is rejected before execution.
- `benchmark/benchmark_engine.py:249-250` explicitly accepts `hybrid`.
- The README also documents `hybrid` datasets and builder targets.

Impact:

- API callers cannot submit the documented mode.
- CLI training config validation would also reject it.
- The codebase is internally inconsistent around a core user-facing feature.

### 3. Algorithms marked `feedback="both"` are treated as incompatible with explicit or implicit datasets

- Severity: High
- Files: `algorithms/__init__.py:27`, `algorithms/__init__.py:139`, `algorithms/__init__.py:153`, `benchmark/benchmark_engine.py:21-24`, `benchmark/benchmark_engine.py:266-268`, `benchmark/benchmark_engine.py:290`, `recommendation/strategy_service.py:26-31`, `recommendation/strategy_service.py:494-500`, `recommendation/strategy_service.py:700-725`, `recommendation/strategy_service.py:809-814`, `tests/test_benchmark_mode_filtering.py:7-25`

Several algorithms are registered with `feedback="both"` such as `SVD`, `Temporal-SVD`, `User-KNN`, `Item-KNN`, `EASE`, `LightFM`, and others. Those should be eligible for either explicit or implicit runs, but current compatibility maps exclude them from both pure modes:

- Benchmark path: `benchmark/benchmark_engine.py` only allows `{"explicit"}` for explicit mode and `{"implicit"}` for implicit mode.
- Serving path: `recommendation/strategy_service.py` uses the same logic when deciding whether a model is recommendation-eligible.
- The existing test file currently asserts this broken behavior instead of catching it.

Verified behavior:

- `RecommendationStrategyService._is_mode_compatible("explicit", "SVD")` returns `False`
- `RecommendationStrategyService._is_mode_compatible("implicit", "SVD")` returns `False`
- `BenchmarkEngine._check_compat({"feedback":"both"}, "explicit", ...)` returns an incompatibility message

Impact:

- Benchmarking can skip strong general-purpose models on explicit and implicit datasets.
- Recommendation options can hide or reject saved models that should be valid.
- Training and serving decisions become materially worse than intended.

### 4. Frontend recommendation `top_n` control conflicts with backend validation

- Severity: Medium
- Files: `frontend/App.jsx:1572`, `frontend/App.jsx:1718`, `api/routes.py:111-122`, `api/routes.py:146-157`

The React app allows any recommendation count from `1` to `100`:

- `frontend/App.jsx:1718` renders `<input type="number" ... min={1} max={100} />`
- `frontend/App.jsx:1572` sends `top_n: Number(form.top_n) || 10`

The backend accepts only `5` or `10`:

- `api/routes.py:111-122` validates `RecommendRequest.top_n`
- `api/routes.py:146-157` validates `BatchRecommendRequest.top_n`

Impact:

- The UI can generate guaranteed `422` errors for valid-looking user input such as `3`, `8`, or `20`.
- This is a direct frontend/backend contract mismatch.

## Verification

Completed:

- `venv\Scripts\python.exe -m unittest tests.test_recommendation_strategy_service tests.test_benchmark_mode_filtering`
- Result: 13 tests passed

Blocked in this environment:

- `node tests\ui_recommendation_ui_logic.test.js`
- Result: failed with a sandbox-level `EPERM` while Node tried to resolve `C:\Users\hi`, so JavaScript test verification is incomplete here

## Recommended Fix Order

1. Fix `main.py` so the module entrypoint calls `main()`.
2. Add `hybrid` back into the global mode validators.
3. Treat `feedback="both"` as compatible with explicit and implicit modes in both benchmark and recommendation paths.
4. Align frontend `top_n` input with backend rules, or relax the backend if broader values are intended.
