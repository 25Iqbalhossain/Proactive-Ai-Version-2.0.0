# How `smart_db_csv_builder` Connects To The Pipeline

`smart_db_csv_builder` is connected to your pipeline as a dataset-building stage before training.

It does **not** call `TrainingPipeline` directly inside Python.
Instead, the connection works like this:

1. The unified FastAPI app mounts both systems together.
2. `smart_db_csv_builder` builds a CSV or JSON dataset from live databases.
3. The frontend downloads the built CSV from the builder job endpoint.
4. The frontend turns that CSV into a browser `File` object.
5. The frontend uploads that file to `/train/file`.
6. The training pipeline loads that file and runs benchmarking, tuning, model saving, and recommendation setup.

## 1. Shared App Wiring

Both parts live in the same FastAPI app in `app_factory.py`.

- Main recommendation/training API:
  - `api.routes`
- Smart DB builder API:
  - `/smart-db-csv/api/connections`
  - `/smart-db-csv/api/schema`
  - `/smart-db-csv/api/build`
  - `/smart-db-csv/api/jobs`

So the builder is not a separate product. It is mounted into the same backend application.

## 2. Builder Side

The builder flow starts in `smart_db_csv_builder/api/routers/build.py`.

When the frontend calls:

```text
POST /smart-db-csv/api/build
```

the backend creates a background job and runs:

```text
smart_db_csv_builder.services.builder.run_build_job(...)
```

That job does this:

1. Validates selected connection IDs.
2. Reads schemas from the registered database connections.
3. Builds a merge plan:
   - manual mode
   - query mode
   - LLM mode
4. Executes queries through `smart_db_csv_builder/services/executor.py`.
5. Merges data into one dataframe.
6. Writes temporary output files:
   - CSV
   - JSON
7. Stores the output file paths in the builder job record.

The output dataset is made available through:

```text
GET /smart-db-csv/api/jobs/{job_id}/download?output_format=csv
```

## 3. Frontend Handoff

The actual connection into the pipeline happens in `frontend/App.jsx`.

After the build job finishes, the frontend does this:

1. Calls the builder download endpoint.
2. Reads the CSV response as a `Blob`.
3. Creates:

```js
new File([blob], 'built_dataset.csv', { type: 'text/csv' })
```

4. Saves that file into React state with `onBuilt(...)`.
5. Automatically moves the user from the Build page to the Train page.

This is the key bridge:

`smart_db_csv_builder` -> downloaded CSV -> browser `File` -> `/train/file`

## 4. Training Pipeline Entry

Training starts from `api/routes.py` at:

```text
POST /train/file
```

That endpoint:

1. Receives the uploaded file.
2. Saves it to a temporary path.
3. Creates a `TrainingConfig`.
4. Runs:

```text
TrainingPipeline(...).run_from_file(temp_path)
```

inside a background job.

The pipeline implementation is in `pipeline/training_pipeline.py`.

It then runs:

1. file loading
2. column detection
3. feedback profiling
4. data cleaning
5. train/test split
6. interaction matrix building
7. benchmark
8. Optuna tuning
9. model saving
10. best-model promotion

## 5. What Happens After Training

After `/train/file` completes, `api/routes.py` updates runtime state:

- `train_df`
- `last_result`
- `serving_im`

Those values are then used by the recommendation and serving endpoints.

So the full chain is:

```text
Database connections
-> schema inspection
-> smart dataset build
-> CSV download
-> /train/file upload
-> TrainingPipeline.run_from_file(...)
-> model registry
-> recommendation/serving endpoints
```

## 6. Important Note

If you were expecting `smart_db_csv_builder` to import and call `TrainingPipeline` directly, that is **not** how this repo is currently wired.

The current integration is:

- unified backend application
- separate API modules
- frontend-controlled handoff between build and train

So the connection is real, but it is an **API/UI workflow connection**, not a direct Python function-to-function connection.

## 7. Main Files To Read

- `app_factory.py`
  - mounts both the builder routers and the core training/recommendation router
- `smart_db_csv_builder/api/routers/build.py`
  - starts dataset build jobs
- `smart_db_csv_builder/services/builder.py`
  - orchestrates schema extraction, planning, and output generation
- `smart_db_csv_builder/services/executor.py`
  - executes queries and writes CSV/JSON output files
- `smart_db_csv_builder/api/routers/jobs.py`
  - exposes the dataset download endpoint
- `frontend/App.jsx`
  - downloads the built CSV and forwards it into training
- `api/routes.py`
  - receives the file at `/train/file` and starts training
- `pipeline/training_pipeline.py`
  - runs the training workflow itself

## 8. Short Answer

`smart_db_csv_builder` connects to your pipeline by generating a training-ready CSV from databases, and the frontend immediately feeds that CSV into the main training API endpoint, which starts `TrainingPipeline.run_from_file(...)`.
