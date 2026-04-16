"""
Serving pipeline for real-time recommendations.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from data_processing.interaction_matrix import InteractionMatrix, InteractionMatrixBuilder
from insights.explainability import ExplainabilityEngine, Explanation
from models.model_loader import ModelLoader
from models.model_registry import ModelRegistry
from recommendation.recommender_engine import RecommendationResult, RecommenderEngine
from utils.logger import get_logger

log = get_logger(__name__)


class ServingPipeline:
    def __init__(self, engine: RecommenderEngine, explainer: ExplainabilityEngine):
        self._engine = engine
        self._explainer = explainer

    @classmethod
    def from_registry(
        cls,
        algorithm: str,
        train_df: Optional[pd.DataFrame] = None,
        item_metadata: Optional[pd.DataFrame] = None,
        registry: Optional[ModelRegistry] = None,
    ) -> "ServingPipeline":
        registry = registry or ModelRegistry()
        loader = ModelLoader(registry)
        loaded = loader.load_promoted(algorithm)

        model = loaded.model
        im: Optional[InteractionMatrix] = None
        if isinstance(model, dict) and "interaction_matrix" in model:
            im = model["interaction_matrix"]
        elif train_df is not None:
            im = InteractionMatrixBuilder().build(train_df)
        else:
            raise RuntimeError(
                "No interaction matrix available. Provide train_df or retrain models with interaction_matrix persisted."
            )

        engine = RecommenderEngine(loader=loader, interaction_matrix=im, default_algorithm=algorithm)
        user_factors, item_factors = cls._extract_latent_factors(model)
        explainer = ExplainabilityEngine(
            im=im,
            user_factors=user_factors,
            item_factors=item_factors,
            item_metadata=item_metadata,
        )

        log.info("ServingPipeline initialized | algo=%s model_id=%s", algorithm, loaded.model_id)
        return cls(engine=engine, explainer=explainer)

    @staticmethod
    def _extract_latent_factors(model) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if isinstance(model, dict):
            inner = model.get("model")
            if inner is not None and hasattr(inner, "user_factors") and hasattr(inner, "item_factors"):
                return np.asarray(inner.user_factors), np.asarray(inner.item_factors)
            if "user_factors" in model and "item_factors" in model:
                return np.asarray(model["user_factors"]), np.asarray(model["item_factors"])
            if "P" in model and "Q" in model:
                return np.asarray(model["P"]), np.asarray(model["Q"])
        elif hasattr(model, "user_factors") and hasattr(model, "item_factors"):
            return np.asarray(model.user_factors), np.asarray(model.item_factors)
        return None, None

    def recommend(
        self,
        user_id: str,
        top_n: int = 10,
        algorithm: Optional[str] = None,
        exclude_seen: bool = True,
    ) -> RecommendationResult:
        return self._engine.recommend(
            user_id=user_id,
            top_n=top_n,
            algorithm=algorithm,
            exclude_seen=exclude_seen,
        )

    def recommend_batch(
        self,
        user_ids: list[str],
        top_n: int = 10,
        algorithm: Optional[str] = None,
    ) -> list[RecommendationResult]:
        return self._engine.recommend_batch(user_ids=user_ids, top_n=top_n, algorithm=algorithm)

    def similar_items(self, item_id: str, top_n: int = 10, algorithm: Optional[str] = None) -> list[dict]:
        return self._engine.similar_items(item_id=item_id, top_n=top_n, algorithm=algorithm)

    def explain(self, user_id: str, item_id: str, score: float = 0.0) -> Explanation:
        return self._explainer.explain(user_id=user_id, item_id=item_id, score=score)

    def explain_recommendations(
        self, user_id: str, result: RecommendationResult
    ) -> list[Explanation]:
        return self._explainer.explain_batch(user_id, result.recommendations)

    def health(self) -> dict:
        return {"status": "ok", "loaded_models": self._engine.loader.list_loaded()}

