from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from recommendation.recommender_engine import RecommendationResult
from recommendation.strategy_service import (
    RecommendationPayload,
    RecommendationStrategyError,
    RecommendationStrategyService,
    WeightedModelInput,
)


def _rec(item_id: str, score: float, rank: int) -> dict:
    return {"item_id": item_id, "score": score, "rank": rank}


class _FakeRegistry:
    def __init__(self, records: list[SimpleNamespace]):
        self._records = sorted(records, key=lambda r: getattr(r, "created_at", 0), reverse=True)

    def list_models(self, algorithm=None, promoted_only=False):
        out = self._records
        if algorithm:
            out = [r for r in out if r.algorithm == algorithm]
        if promoted_only:
            out = [r for r in out if getattr(r, "promoted", False)]
        return out

    def get(self, model_id: str):
        for record in self._records:
            if record.model_id == model_id:
                return record
        return None

    def latest_for_algorithm(self, algorithm: str):
        matches = [r for r in self._records if r.algorithm == algorithm]
        return matches[0] if matches else None

    def get_promoted(self, algorithm: str):
        for record in self._records:
            if record.algorithm == algorithm and getattr(record, "promoted", False):
                return record
        return None


class _FakeLoader:
    def __init__(self, records: list[SimpleNamespace]):
        self._by_id = {r.model_id: r for r in records}

    def load_by_id(self, model_id: str):
        record = self._by_id.get(model_id)
        if not record:
            raise KeyError(f"model '{model_id}' not found")
        return SimpleNamespace(model_id=record.model_id, algorithm=record.algorithm, model={})


class _FakeRecommender:
    def __init__(self, outputs: dict[str, list[dict] | Exception]):
        self.outputs = outputs

    def recommend_with_loaded_model(self, user_id, loaded, top_n, algorithm_label, exclude_seen=True):
        result = self.outputs.get(loaded.model_id)
        if isinstance(result, Exception):
            raise result
        rows = list(result or [])[:top_n]
        return RecommendationResult(
            user_id=str(user_id),
            algorithm=algorithm_label or loaded.algorithm,
            model_id=loaded.model_id,
            recommendations=rows,
            top_n=top_n,
            generated_at=0.0,
        )


class _FakeReport:
    def __init__(self, rows, primary_metric="NDCG@K", primary_metric_direction="maximize"):
        self._rows = list(rows)
        self.primary_metric = primary_metric
        self.primary_metric_direction = primary_metric_direction

    def leaderboard(self):
        return pd.DataFrame(self._rows)


class RecommendationStrategyServiceTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            SimpleNamespace(
                algorithm="SVD",
                model_id="m_svd",
                created_at=10,
                promoted=True,
                is_implicit=False,
                metrics={"NDCG@K": 0.91, "Composite Score": 0.95},
            ),
            SimpleNamespace(
                algorithm="ALS",
                model_id="m_als",
                created_at=9,
                promoted=True,
                is_implicit=True,
                metrics={"NDCG@K": 0.72, "Composite Score": 0.81},
            ),
            SimpleNamespace(
                algorithm="BPR",
                model_id="m_bpr",
                created_at=8,
                promoted=True,
                is_implicit=True,
                metrics={"NDCG@K": 0.68, "Composite Score": 0.76},
            ),
            SimpleNamespace(
                algorithm="Temporal-SVD",
                model_id="m_temp",
                created_at=7,
                promoted=False,
                is_implicit=False,
                metrics={"NDCG@K": 0.63, "Composite Score": 0.7},
            ),
        ]
        self.last_result = SimpleNamespace(
            resolved_mode="hybrid",
            best_model_id="m_svd",
            best_algorithm="SVD",
            top_model_recommendations=[
                {"rank": 1, "algorithm": "SVD", "model_id": "m_svd", "selection_score_pct": 96.0, "summary": "SVD leads."},
                {"rank": 2, "algorithm": "ALS", "model_id": "m_als", "selection_score_pct": 78.0, "summary": "ALS is competitive."},
                {"rank": 3, "algorithm": "BPR", "model_id": "m_bpr", "selection_score_pct": 72.0, "summary": "BPR remains viable."},
                {"rank": 4, "algorithm": "Temporal-SVD", "model_id": "m_temp", "selection_score_pct": 64.0, "summary": "Temporal-SVD fits temporal data."},
            ],
            model_selection_policy={
                "selection_type": "top_ranked_models",
                "selected_count": 4,
                "selected_models": [
                    {
                        "rank": 1,
                        "algorithm": "SVD",
                        "model_id": "m_svd",
                        "selection_score_pct": 96.0,
                        "metric_name": "NDCG@K",
                        "metric_value": 0.91,
                        "summary": "SVD leads.",
                    },
                    {
                        "rank": 2,
                        "algorithm": "ALS",
                        "model_id": "m_als",
                        "selection_score_pct": 78.0,
                        "metric_name": "NDCG@K",
                        "metric_value": 0.72,
                        "summary": "ALS is competitive.",
                    },
                    {
                        "rank": 3,
                        "algorithm": "BPR",
                        "model_id": "m_bpr",
                        "selection_score_pct": 72.0,
                        "metric_name": "NDCG@K",
                        "metric_value": 0.68,
                        "summary": "BPR remains viable.",
                    },
                    {
                        "rank": 4,
                        "algorithm": "Temporal-SVD",
                        "model_id": "m_temp",
                        "selection_score_pct": 64.0,
                        "metric_name": "NDCG@K",
                        "metric_value": 0.63,
                        "summary": "Temporal-SVD fits temporal data.",
                    },
                ],
                "reason": "SVD is clearly ahead, so the API returns one recommended model by default. Showing the top 4 ranked models so you can compare why each one scored well.",
                "display_title": "Top 4 model recommendations and reasons",
                "recommended_strategy": "single_model",
                "serving_selected_count": 1,
                "serving_selected_models": [
                    {
                        "rank": 1,
                        "algorithm": "SVD",
                        "model_id": "m_svd",
                        "selection_score_pct": 96.0,
                        "metric_name": "NDCG@K",
                        "metric_value": 0.91,
                        "summary": "SVD leads.",
                    }
                ],
                "serving_reason": "SVD is clearly ahead, so the API returns one recommended model by default.",
            },
            report=_FakeReport(
                [
                    {"Rank": 1, "Algorithm": "SVD", "NDCG@K": 0.91, "Composite Score": 0.95},
                    {"Rank": 2, "Algorithm": "ALS", "NDCG@K": 0.72, "Composite Score": 0.81},
                    {"Rank": 3, "Algorithm": "BPR", "NDCG@K": 0.68, "Composite Score": 0.76},
                    {"Rank": 4, "Algorithm": "Temporal-SVD", "NDCG@K": 0.63, "Composite Score": 0.7},
                ]
            ),
        )
        self.base_outputs = {
            "m_svd": [_rec("A", 9.0, 1), _rec("B", 8.0, 2), _rec("C", 7.0, 3)],
            "m_als": [_rec("B", 3.0, 1), _rec("C", 2.0, 2), _rec("D", 1.0, 3)],
            "m_bpr": [_rec("A", 5.0, 1), _rec("D", 4.0, 2), _rec("E", 3.0, 3)],
            "m_temp": [_rec("A", 2.0, 1), _rec("E", 1.0, 2)],
        }

    def _service(self, outputs=None):
        outputs = outputs or self.base_outputs
        return RecommendationStrategyService(
            registry=_FakeRegistry(self.records),
            loader=_FakeLoader(self.records),
            recommender=_FakeRecommender(outputs),
        )

    def test_single_model_recommendation(self):
        svc = self._service()
        response = svc.recommend(
            RecommendationPayload(user_id="u1", top_n=3, strategy="single_model", algorithm="SVD"),
            last_result=self.last_result,
        )
        self.assertEqual(response["strategy"], "single_model")
        self.assertEqual(len(response["models_used"]), 1)
        self.assertEqual(response["models_used"][0]["algorithm"], "SVD")
        self.assertEqual(response["recommendations"][0]["item_id"], "A")
        self.assertEqual(response["recommendations"][0]["contributions"][0]["share_pct"], 100.0)

    def test_ensemble_weighted_valid_request(self):
        svc = self._service()
        response = svc.recommend(
            RecommendationPayload(
                user_id="u1",
                top_n=3,
                strategy="ensemble_weighted",
                models=[
                    WeightedModelInput(algorithm="SVD", weight=50),
                    WeightedModelInput(algorithm="ALS", weight=30),
                    WeightedModelInput(algorithm="BPR", weight=20),
                ],
            ),
            last_result=self.last_result,
        )
        self.assertEqual(response["strategy"], "ensemble_weighted")
        self.assertEqual(response["recommendations"][0]["item_id"], "A")
        self.assertAlmostEqual(response["recommendations"][0]["final_score"], 0.7, places=4)
        self.assertEqual(len(response["models_used"]), 3)
        self.assertTrue(any(c["algorithm"] == "SVD" for c in response["recommendations"][0]["contributions"]))

    def test_duplicate_model_rejection(self):
        svc = self._service()
        with self.assertRaises(RecommendationStrategyError):
            svc.recommend(
                RecommendationPayload(
                    user_id="u1",
                    top_n=5,
                    strategy="ensemble_weighted",
                    models=[
                        WeightedModelInput(algorithm="SVD", weight=0.6),
                        WeightedModelInput(algorithm="SVD", weight=0.4),
                    ],
                ),
                last_result=self.last_result,
            )

    def test_invalid_weight_rejection(self):
        svc = self._service()
        with self.assertRaises(RecommendationStrategyError):
            svc.recommend(
                RecommendationPayload(
                    user_id="u1",
                    top_n=5,
                    strategy="ensemble_weighted",
                    models=[
                        WeightedModelInput(algorithm="SVD", weight=1.0),
                        WeightedModelInput(algorithm="ALS", weight=0.0),
                    ],
                ),
                last_result=self.last_result,
            )

    def test_auto_normalization(self):
        svc = self._service()
        response = svc.recommend(
            RecommendationPayload(
                user_id="u1",
                top_n=3,
                strategy="ensemble_weighted",
                auto_normalize_weights=True,
                models=[
                    WeightedModelInput(algorithm="SVD", weight=2.0),
                    WeightedModelInput(algorithm="ALS", weight=1.0),
                ],
            ),
            last_result=self.last_result,
        )
        weights = {m["algorithm"]: m["normalized_weight"] for m in response["models_used"]}
        self.assertTrue(response["weight_normalization_applied"])
        self.assertAlmostEqual(weights["SVD"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(weights["ALS"], 1.0 / 3.0, places=6)

    def test_partial_model_failure_continues_with_rebalanced_weights(self):
        records = self.records + [SimpleNamespace(algorithm="Temporal-SVD", model_id="m_temp_fail", created_at=11, promoted=False)]
        outputs = dict(self.base_outputs)
        outputs["m_temp_fail"] = RuntimeError("inference failure")
        svc = RecommendationStrategyService(
            registry=_FakeRegistry(records),
            loader=_FakeLoader(records),
            recommender=_FakeRecommender(outputs),
        )
        response = svc.recommend(
            RecommendationPayload(
                user_id="u1",
                top_n=3,
                strategy="ensemble_weighted",
                models=[
                    WeightedModelInput(algorithm="SVD", model_id="m_svd", weight=0.5),
                    WeightedModelInput(algorithm="ALS", model_id="m_als", weight=0.3),
                    WeightedModelInput(algorithm="Temporal-SVD", model_id="m_temp_fail", weight=0.2),
                ],
            ),
            last_result=self.last_result,
        )
        self.assertEqual(len(response["models_used"]), 2)
        self.assertTrue(response["weight_rebalanced_after_runtime_failures"])
        self.assertTrue(any("failed" in w.lower() for w in response["warnings"]))

    def test_no_predictions_from_one_model_continues(self):
        outputs = dict(self.base_outputs)
        outputs["m_als"] = []
        svc = self._service(outputs=outputs)
        response = svc.recommend(
            RecommendationPayload(
                user_id="u1",
                top_n=3,
                strategy="ensemble_weighted",
                models=[
                    WeightedModelInput(algorithm="SVD", weight=0.5),
                    WeightedModelInput(algorithm="ALS", weight=0.3),
                    WeightedModelInput(algorithm="BPR", weight=0.2),
                ],
            ),
            last_result=self.last_result,
        )
        self.assertEqual(len(response["models_used"]), 2)
        self.assertTrue(any("returned no recommendations" in w for w in response["warnings"]))

    def test_recommendation_options_include_feedback_mode_after_training(self):
        svc = self._service()
        options = svc.recommendation_options(last_result=self.last_result)

        self.assertEqual(options["best_promoted_model"]["model_id"], "m_svd")
        self.assertEqual(options["single_model_options"][0]["feedback_mode"], "both")
        self.assertTrue(options["has_recommendation_models"])
        self.assertEqual(options["model_id_map"]["SVD"], "m_svd")
        self.assertEqual(options["selection_policy"]["selection_type"], "top_ranked_models")
        self.assertEqual(options["recommended_models"][0]["algorithm"], "SVD")
        self.assertEqual(len(options["recommended_models"]), 1)
        self.assertEqual(options["supported_model_count"], 4)
        self.assertEqual(options["ranked_model_count"], 1)
        self.assertEqual(options["ranked_models"][0]["algorithm"], "SVD")
        self.assertIn("selection score", options["best_model_explanation"].lower())
        self.assertIn("comparison_to_next", options["supported_models"][0])

    def test_recommendation_options_can_return_requested_top_n_ranked_models(self):
        svc = self._service()
        options = svc.recommendation_options(last_result=self.last_result, top_n_models=3)

        self.assertEqual(options["ranked_model_limit"], 3)
        self.assertEqual(len(options["ranked_models"]), 3)
        self.assertEqual(
            [row["algorithm"] for row in options["ranked_models"]],
            ["SVD", "ALS", "BPR"],
        )
        self.assertIn("Top 1", options["ranked_models"][0]["reason"])
        self.assertIn("selection score", options["ranked_models"][0]["comparison_to_next"])

    def test_serialise_training_result_includes_recommendation_options(self):
        from api import routes

        svc = self._service()
        result = SimpleNamespace(
            report=self.last_result.report,
            best_algorithm="SVD",
            best_params={"factors": 32},
            best_model_id="m_svd",
            elapsed_s=12.34,
            all_model_ids=["m_svd", "m_als", "m_bpr"],
            resolved_mode="hybrid",
            feedback_profile={"detected_mode": "hybrid"},
            ranking_logic={"primary_metric": "NDCG@K"},
            optuna_note="adaptive",
            optuna_policy={
                "summary": "Optuna uses adaptive per-algorithm budgets.",
                "top_k_definition": "Top-K=10 evaluates the top 10 ranked items.",
            },
            top_model_recommendations=list(self.last_result.top_model_recommendations),
            model_selection_policy=dict(self.last_result.model_selection_policy),
            tuning_results=[
                SimpleNamespace(
                    algorithm="SVD",
                    best_value=0.91,
                    metric_name="NDCG@K",
                    best_params={"factors": 32},
                    n_trials=5,
                    status="ok",
                    fallback_reason="",
                    trial_budget="adaptive(5)",
                )
            ],
        )

        with patch.object(routes, "_strategy_service", svc):
            payload = routes._serialise_training_result(result)

        self.assertEqual(payload["best_model_id"], "m_svd")
        self.assertIn("recommendation_options", payload)
        self.assertEqual(payload["recommendation_options"]["best_promoted_model"]["model_id"], "m_svd")
        self.assertEqual(payload["recommendation_options"]["single_model_options"][0]["feedback_mode"], "both")
        self.assertEqual(payload["model_selection_policy"]["selection_type"], "top_ranked_models")
        self.assertEqual(payload["top_model_candidates"][0]["algorithm"], "SVD")
        self.assertEqual(len(payload["top_model_candidates"]), 4)
        self.assertIn("optuna_policy", payload)


if __name__ == "__main__":
    unittest.main()
