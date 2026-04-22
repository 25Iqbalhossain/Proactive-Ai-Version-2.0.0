"""
FastAPI route definitions.

Public:
- /health
- /session/status
- /algorithms
- /auth/login
- /train/file
- /train/sql
- /jobs/{job_id}

Protected (Bearer token):
- /auth/me
- /recommend
- /recommend/batch
- /similar/{item_id}
- /explain/{user_id}/{item_id}
- /models
- /models/{model_id}/promote
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from algorithms import ALGORITHM_REGISTRY
from api.auth import create_access_token, require_auth, validate_credentials
from config.settings import (
    ALGORITHM_MODES,
    DEFAULT_ALGORITHM_MODE,
    OPTUNA_TRIALS_AUTO,
    TOP_K,
    TOP_K_ALLOWED,
    TOP_MODEL_ALLOWED,
    TOP_N_MODELS,
)
from models.model_registry import ModelRegistry
from models.model_loader import ModelLoader
from pipeline.serving_pipeline import ServingPipeline
from pipeline.training_pipeline import TrainingConfig, TrainingPipeline
from recommendation.recommender_engine import RecommenderEngine
from recommendation.strategy_service import (
    RecommendationPayload,
    RecommendationStrategyError,
    RecommendationStrategyService,
    WeightedModelInput,
    RECOMMEND_STRATEGIES,
)
from data_processing.interaction_matrix import InteractionMatrixBuilder
from utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter()
_registry = ModelRegistry()
_executor = ThreadPoolExecutor(max_workers=4)
_model_loader = ModelLoader(_registry)
_direct_recommender = RecommenderEngine(_model_loader, interaction_matrix=None)
_strategy_service = RecommendationStrategyService(_registry, _model_loader, _direct_recommender)

# In-memory runtime state
_jobs: dict[str, dict] = {}
_runtime_state: dict[str, object] = {
    "train_df": None,
    "last_result": None,
    "serving_im": None,
}
_serving_cache: dict[str, ServingPipeline] = {}


class LoginRequest(BaseModel):
    username: str
    password: str


class WeightedModelRequest(BaseModel):
    algorithm: str
    weight: float
    model_id: Optional[str] = None

    @field_validator("algorithm")
    @classmethod
    def _validate_algorithm(cls, value: str) -> str:
        v = (value or "").strip()
        if not v:
            raise ValueError("algorithm is required")
        return v

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("weight must be positive")
        return value


class RecommendRequest(BaseModel):
    user_id: str
    top_n: int = 10
    strategy: Optional[str] = None
    model_id: Optional[str] = None
    algorithm: Optional[str] = None
    models: list[WeightedModelRequest] = Field(default_factory=list)
    auto_normalize_weights: bool = True

    @field_validator("top_n")
    @classmethod
    def _validate_top_n(cls, value: int) -> int:
        if value not in TOP_K_ALLOWED:
            raise ValueError(f"top_n must be one of {TOP_K_ALLOWED}")
        return value

    @field_validator("user_id")
    @classmethod
    def _validate_user_id(cls, value: str) -> str:
        v = (value or "").strip()
        if not v:
            raise ValueError("user_id is required")
        return v

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        v = value.strip().lower()
        if v not in RECOMMEND_STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(RECOMMEND_STRATEGIES)}")
        return v


class BatchRecommendRequest(BaseModel):
    user_ids: list[str] = Field(default_factory=list)
    top_n: int = 10
    strategy: Optional[str] = None
    model_id: Optional[str] = None
    algorithm: Optional[str] = None
    models: list[WeightedModelRequest] = Field(default_factory=list)
    auto_normalize_weights: bool = True

    @field_validator("top_n")
    @classmethod
    def _validate_top_n(cls, value: int) -> int:
        if value not in TOP_K_ALLOWED:
            raise ValueError(f"top_n must be one of {TOP_K_ALLOWED}")
        return value

    @field_validator("user_ids")
    @classmethod
    def _validate_user_ids(cls, value: list[str]) -> list[str]:
        if not value:
            return value
        cleaned = [v.strip() for v in value if v and v.strip()]
        if len(cleaned) != len(value):
            raise ValueError("user_ids must not contain empty values")
        return cleaned

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        v = value.strip().lower()
        if v not in RECOMMEND_STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(RECOMMEND_STRATEGIES)}")
        return v


class SQLTrainRequest(BaseModel):
    dialect: str
    host: str
    port: int
    database: str
    username: str
    password: str
    sql: str
    top_k: int = TOP_K
    n_trials: int = OPTUNA_TRIALS_AUTO
    top_models: int = TOP_N_MODELS
    algorithm_mode: str = DEFAULT_ALGORITHM_MODE

    @field_validator("top_k")
    @classmethod
    def _validate_top_k(cls, value: int) -> int:
        if value not in TOP_K_ALLOWED:
            raise ValueError(f"top_k must be one of {TOP_K_ALLOWED}")
        return value

    @field_validator("top_models")
    @classmethod
    def _validate_top_models(cls, value: int) -> int:
        if value not in TOP_MODEL_ALLOWED:
            raise ValueError(f"top_models must be one of {TOP_MODEL_ALLOWED}")
        return value

    @field_validator("algorithm_mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        v = (value or "").lower().strip()
        if v not in ALGORITHM_MODES:
            raise ValueError(f"algorithm_mode must be one of {ALGORITHM_MODES}")
        return v


def _submit_job(fn, *args, **kwargs) -> str:
    job_id = str(uuid.uuid4())[:12]
    _jobs[job_id] = {
        "status": "pending",
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
    }

    def _run():
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = time.time()
        try:
            result = fn(*args, **kwargs)
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = result
            _jobs[job_id]["finished_at"] = time.time()
        except Exception as e:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["finished_at"] = time.time()
            log.error("Job %s failed: %s", job_id, e, exc_info=True)

    _executor.submit(_run)
    return job_id


@router.get("/health", tags=["System"])
def health():
    return {"status": "ok", "timestamp": time.time()}


@router.get("/session/status", tags=["System"])
def session_status():
    train_df = _runtime_state.get("train_df")
    last_result = _runtime_state.get("last_result")
    options = _strategy_service.recommendation_options(last_result=last_result)
    return {
        "dataset_loaded": train_df is not None,
        "last_best_algorithm": getattr(last_result, "best_algorithm", None) if last_result else None,
        "last_mode": getattr(last_result, "resolved_mode", None) if last_result else None,
        "cached_serving_algorithms": list(_serving_cache.keys()),
        "recommendation_options": options,
    }


@router.get("/algorithms", tags=["System"])
def list_algorithms():
    return {
        "algorithms": [
            {
                "name": name,
                "feedback": meta.get("feedback"),
                "domain": meta.get("domain"),
                "description": meta.get("description"),
                "training_speed": meta.get("training_speed"),
                "scalability": meta.get("scalability"),
                "robustness": meta.get("robustness"),
                "interpretability": meta.get("interpretability"),
                "production_readiness": meta.get("production_readiness"),
            }
            for name, meta in ALGORITHM_REGISTRY.items()
        ]
    }


@router.post("/auth/login", tags=["Auth"])
def login(req: LoginRequest):
    if not validate_credentials(req.username, req.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(subject=req.username)
    return {"access_token": token, "token_type": "bearer"}


@router.get("/auth/me", tags=["Auth"])
def whoami(payload: dict = Depends(require_auth)):
    return {"user": payload.get("sub"), "exp": payload.get("exp")}


@router.post("/train/file", tags=["Training"])
async def train_from_file(
    file: UploadFile = File(..., description="CSV/Excel/JSON/Parquet file"),
    top_k: int = Form(TOP_K),
    n_trials: int = Form(OPTUNA_TRIALS_AUTO),
    top_models: int = Form(TOP_N_MODELS),
    algorithm_mode: str = Form(DEFAULT_ALGORITHM_MODE),
    format: str = Form("auto"),
):
    _validate_top_choice("top_k", top_k, TOP_K_ALLOWED)
    _validate_top_choice("top_models", top_models, TOP_MODEL_ALLOWED)
    _validate_algorithm_mode(algorithm_mode)

    suffix = Path(file.filename).suffix if file.filename else ""
    temp_path = Path(tempfile.mkstemp(prefix="proactive_upload_", suffix=suffix)[1])
    try:
        file.file.seek(0)
        with open(temp_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
    finally:
        await file.close()

    config = TrainingConfig(
        top_k=top_k,
        n_tuning_trials=n_trials,
        top_model_count=top_models,
        algorithm_mode=algorithm_mode.lower(),
    )

    def _task():
        try:
            pipeline = TrainingPipeline(config)
            result = pipeline.run_from_file(str(temp_path), file_format=format)
            _runtime_state["train_df"] = result.report.train.copy()
            _runtime_state["last_result"] = result
            _runtime_state["serving_im"] = InteractionMatrixBuilder().build(result.report.train)
            _direct_recommender.im = _runtime_state["serving_im"]
            _serving_cache.clear()
            return _serialise_training_result(result)
        finally:
            try:
                if temp_path.exists():
                    os.remove(temp_path)
            except Exception:
                pass

    job_id = _submit_job(_task)
    return JSONResponse(status_code=202, content={"job_id": job_id, "poll_url": f"/jobs/{job_id}"})


@router.post("/train/sql", tags=["Training"])
def train_from_sql(req: SQLTrainRequest):
    from config.database import SQLConfig

    sql_config = SQLConfig(
        dialect=req.dialect,
        host=req.host,
        port=req.port,
        database=req.database,
        username=req.username,
        password=req.password,
    )
    config = TrainingConfig(
        top_k=req.top_k,
        n_tuning_trials=req.n_trials,
        top_model_count=req.top_models,
        algorithm_mode=req.algorithm_mode,
    )

    def _task():
        pipeline = TrainingPipeline(config)
        result = pipeline.run_from_sql(sql_config, req.sql)
        _runtime_state["train_df"] = result.report.train.copy()
        _runtime_state["last_result"] = result
        _runtime_state["serving_im"] = InteractionMatrixBuilder().build(result.report.train)
        _direct_recommender.im = _runtime_state["serving_im"]
        _serving_cache.clear()
        return _serialise_training_result(result)

    job_id = _submit_job(_task)
    return JSONResponse(status_code=202, content={"job_id": job_id, "poll_url": f"/jobs/{job_id}"})


@router.get("/jobs/{job_id}", tags=["Jobs"])
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(content=_sanitise(job))


@router.get("/recommend/options", tags=["Recommendations"])
def recommendation_options(_: dict = Depends(require_auth)):
    last_result = _runtime_state.get("last_result")
    options = _strategy_service.recommendation_options(last_result=last_result)
    options["available_algorithms"] = sorted(ALGORITHM_REGISTRY.keys())
    return _sanitise(options)


@router.post("/recommend", tags=["Recommendations"])
def get_recommendations(req: RecommendRequest, _: dict = Depends(require_auth)):
    payload = RecommendationPayload(
        user_id=req.user_id,
        top_n=req.top_n,
        strategy=req.strategy,
        model_id=req.model_id,
        algorithm=req.algorithm,
        models=[
            WeightedModelInput(algorithm=m.algorithm, weight=m.weight, model_id=m.model_id)
            for m in (req.models or [])
        ],
        auto_normalize_weights=req.auto_normalize_weights,
    )
    try:
        return _sanitise(_strategy_service.recommend(payload=payload, last_result=_runtime_state.get("last_result")))
    except RecommendationStrategyError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.error("Recommendation request failed: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/recommend/batch", tags=["Recommendations"])
def get_batch_recommendations(req: BatchRecommendRequest, _: dict = Depends(require_auth)):
    if not req.user_ids:
        raise HTTPException(status_code=400, detail="user_ids cannot be empty")
    results = []
    errors = []
    for user_id in req.user_ids:
        payload = RecommendationPayload(
            user_id=user_id,
            top_n=req.top_n,
            strategy=req.strategy,
            model_id=req.model_id,
            algorithm=req.algorithm,
            models=[
                WeightedModelInput(algorithm=m.algorithm, weight=m.weight, model_id=m.model_id)
                for m in (req.models or [])
            ],
            auto_normalize_weights=req.auto_normalize_weights,
        )
        try:
            results.append(
                _strategy_service.recommend(payload=payload, last_result=_runtime_state.get("last_result"))
            )
        except RecommendationStrategyError as e:
            errors.append({"user_id": user_id, "error": str(e)})
        except Exception as e:
            errors.append({"user_id": user_id, "error": str(e)})

    if not results:
        raise HTTPException(status_code=422, detail="No batch recommendation results could be generated.")
    return _sanitise({"results": results, "errors": errors})


@router.get("/similar/{item_id}", tags=["Recommendations"])
def similar_items(item_id: str, top_n: int = 10, algorithm: Optional[str] = None, _: dict = Depends(require_auth)):
    _validate_top_choice("top_n", top_n, TOP_K_ALLOWED)
    algo = algorithm or _default_algorithm()
    pipeline = _get_serving_pipeline(algo)
    return {"algorithm": algo, "item_id": item_id, "similar": pipeline.similar_items(item_id, top_n=top_n, algorithm=algo)}


@router.get("/explain/{user_id}/{item_id}", tags=["Recommendations"])
def explain(user_id: str, item_id: str, score: float = 0.0, algorithm: Optional[str] = None, _: dict = Depends(require_auth)):
    algo = algorithm or _default_algorithm()
    pipeline = _get_serving_pipeline(algo)
    exp = pipeline.explain(user_id=user_id, item_id=item_id, score=score)
    return _sanitise(exp.__dict__)


@router.get("/models", tags=["Models"])
def list_models(algorithm: Optional[str] = None, _: dict = Depends(require_auth)):
    records = _registry.list_models(algorithm=algorithm)
    return {"models": [r.to_dict() for r in records]}


@router.post("/models/{model_id}/promote", tags=["Models"])
def promote_model(model_id: str, _: dict = Depends(require_auth)):
    try:
        _registry.promote(model_id)
        _serving_cache.clear()
        return {"status": "promoted", "model_id": model_id}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _get_serving_pipeline(algorithm: str) -> ServingPipeline:
    if algorithm in _serving_cache:
        return _serving_cache[algorithm]

    train_df = _runtime_state.get("train_df")
    try:
        if isinstance(train_df, pd.DataFrame):
            sp = ServingPipeline.from_registry(algorithm=algorithm, train_df=train_df)
        else:
            sp = ServingPipeline.from_registry(algorithm=algorithm, train_df=None)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Serving pipeline is not ready for algorithm '{algorithm}': {e}. "
                "Train/promote a model first."
            ),
        )
    _serving_cache[algorithm] = sp
    return sp


def _default_algorithm() -> str:
    last_result = _runtime_state.get("last_result")
    if last_result and getattr(last_result, "best_algorithm", None):
        return str(last_result.best_algorithm)

    promoted = _registry.list_models(promoted_only=True)
    if not promoted:
        raise HTTPException(status_code=404, detail="No promoted models found. Train and promote first.")
    promoted.sort(key=lambda r: r.created_at, reverse=True)
    return promoted[0].algorithm


def _validate_top_choice(name: str, value: int, allowed: tuple[int, ...]) -> None:
    if int(value) not in allowed:
        raise HTTPException(status_code=422, detail=f"{name} must be one of {allowed}")


def _validate_algorithm_mode(value: str) -> None:
    if (value or "").lower().strip() not in ALGORITHM_MODES:
        raise HTTPException(status_code=422, detail=f"algorithm_mode must be one of {ALGORITHM_MODES}")


def _sanitise(obj):
    import math
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    try:
        import numpy as np
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
    except Exception:
        pass
    return obj


def _serialise_training_result(result) -> dict:
    lb = result.report.leaderboard()
    rows = lb.to_dict(orient="records") if not lb.empty else []
    payload = {
        "best_algorithm": result.best_algorithm,
        "best_params": result.best_params,
        "best_model_id": result.best_model_id,
        "elapsed_s": round(result.elapsed_s, 2),
        "all_model_ids": getattr(result, "all_model_ids", []),
        "leaderboard": rows,
        "resolved_mode": result.resolved_mode,
        "feedback_profile": result.feedback_profile,
        "ranking_logic": result.ranking_logic,
        "optuna_note": result.optuna_note,
        "optuna_policy": getattr(result, "optuna_policy", {}),
        "top_model_recommendations": result.top_model_recommendations,
        "model_selection_policy": getattr(result, "model_selection_policy", {}),
        "recommendation_options": _strategy_service.recommendation_options(last_result=result),
        "top_model_candidates": (
            (getattr(result, "model_selection_policy", {}) or {}).get("selected_models")
            or (result.top_model_recommendations or [])[:5]
        ),
        "tuning": [
            {
                "algorithm": t.algorithm,
                "best_value": round(t.best_value, 4) if isinstance(t.best_value, (int, float)) else None,
                "metric": t.metric_name,
                "best_params": t.best_params,
                "n_trials": t.n_trials,
                "status": t.status,
                "fallback_reason": t.fallback_reason,
                "trial_budget": t.trial_budget,
            }
            for t in result.tuning_results
        ],
    }
    return _sanitise(payload)
