"""
Strategy-aware recommendation orchestration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from algorithms import ALGORITHM_REGISTRY
from models.model_loader import LoadedModel, ModelLoader
from models.model_registry import ModelRecord, ModelRegistry
from recommendation.recommender_engine import RecommenderEngine
from utils.logger import get_logger

log = get_logger(__name__)

RECOMMEND_STRATEGIES = {"best_promoted_model", "single_model", "ensemble_weighted"}
_METRIC_PRIORITY = (
    "NDCG@K",
    "RMSE",
    "MAP@K",
    "Precision@K",
    "Recall@K",
    "Composite Score",
)
_MODE_TO_ALLOWED_FEEDBACK = {
    "explicit": {"explicit", "both"},
    "implicit": {"implicit", "both"},
    "hybrid": {"explicit", "implicit", "both"},
    "auto": {"explicit", "implicit", "both"},
}


@dataclass
class WeightedModelInput:
    weight: float
    model_id: Optional[str] = None
    algorithm: Optional[str] = None
    label: Optional[str] = None
    allow_algorithm_fallback: bool = False


@dataclass
class RecommendationPayload:
    user_id: str
    top_n: int
    strategy: Optional[str] = None
    model_id: Optional[str] = None
    algorithm: Optional[str] = None
    label: Optional[str] = None
    models: list[WeightedModelInput] | None = None
    auto_normalize_weights: bool = True
    allow_algorithm_fallback: bool = False


class RecommendationStrategyError(ValueError):
    def __init__(self, code: str, message: str, **details):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, **self.details}


class RecommendationStrategyService:
    def __init__(
        self,
        registry: ModelRegistry,
        loader: ModelLoader,
        recommender: RecommenderEngine,
    ):
        self.registry = registry
        self.loader = loader
        self.recommender = recommender

    def recommend(
        self,
        payload: RecommendationPayload,
        last_result=None,
    ) -> dict:
        strategy = self._resolve_strategy(payload)
        mode = getattr(last_result, "resolved_mode", None) if last_result else None
        prepared = self._prepare_models(payload=payload, strategy=strategy, last_result=last_result, mode=mode)
        selected_models = prepared["models"]
        warnings = list(prepared["warnings"])

        candidate_top_n = max(int(payload.top_n) * 5, 50)
        model_outputs, runtime_issues = self._run_models(
            selected_models=selected_models,
            user_id=payload.user_id,
            top_n=candidate_top_n,
        )
        warnings.extend(runtime_issues)

        if strategy == "single_model" and not model_outputs:
            issue = runtime_issues[0] if runtime_issues else {}
            raise RecommendationStrategyError(
                issue.get("code", "NO_PREDICTIONS_FOR_USER"),
                issue.get("message", "The selected model did not produce recommendations for this user."),
                strategy=strategy,
                user_id=payload.user_id,
                requested_model_id=payload.model_id,
                requested_algorithm=payload.algorithm,
            )

        if strategy == "ensemble_weighted" and len(model_outputs) < 2:
            raise RecommendationStrategyError(
                "INSUFFICIENT_PREDICTIVE_MODELS",
                "Fewer than two selected models produced predictions for ensemble_weighted.",
                strategy=strategy,
                requested_model_count=len(selected_models),
                predictive_model_count=len(model_outputs),
                warnings=runtime_issues,
            )

        if not model_outputs:
            raise RecommendationStrategyError(
                "NO_MODELS_AVAILABLE",
                "No valid models were available to serve recommendations.",
                strategy=strategy,
            )

        weight_rebalanced = False
        weight_rebalanced_note = ""
        if strategy == "ensemble_weighted":
            model_outputs, weight_rebalanced, weight_rebalanced_note = self._rebalance_runtime_weights(
                model_outputs,
                had_drops=len(model_outputs) != len(selected_models),
            )
            if weight_rebalanced and weight_rebalanced_note:
                warnings.append(self._warning("RUNTIME_WEIGHT_REBALANCED", weight_rebalanced_note))

        recommendations = (
            self._aggregate_weighted(model_outputs, top_n=payload.top_n)
            if strategy == "ensemble_weighted"
            else self._single_model_result(model_outputs[0], top_n=payload.top_n)
        )
        contribution_breakdown = self._build_contribution_breakdown(recommendations)

        response = {
            "strategy": strategy,
            "user_id": payload.user_id,
            "top_n": payload.top_n,
            "weight_normalization_applied": prepared["weight_normalization_applied"],
            "weight_normalization_note": prepared["weight_normalization_note"],
            "normalization_strategy": "min_max_per_model",
            "weight_rebalanced_after_runtime_failures": weight_rebalanced,
            "weight_rebalanced_note": weight_rebalanced_note,
            "models_used": [
                {
                    "model_id": model["model_id"],
                    "algorithm": model["algorithm"],
                    "label": model["label"],
                    "promoted": model["promoted"],
                    "metric_name": model["metric_name"],
                    "metric_value": model["metric_value"],
                    "feedback_mode": model["feedback_mode"],
                    "status": model["status"],
                    "input_weight": model["input_weight"],
                    "normalized_weight": model["normalized_weight"],
                    "served_algorithm": model["served_algorithm"],
                }
                for model in model_outputs
            ],
            "contribution_breakdown": contribution_breakdown,
            "recommendations": recommendations,
            "warnings": warnings,
        }
        if len(model_outputs) == 1:
            response["model_id"] = model_outputs[0]["model_id"]
            response["algorithm"] = model_outputs[0]["algorithm"]
        else:
            response["model_id"] = None
            response["algorithm"] = "ensemble"
        return response

    def recommendation_options(self, last_result=None) -> dict:
        option_rows = self._build_recommendation_option_rows(last_result=last_result)
        best_promoted_model = self._best_promoted_option(last_result=last_result, option_rows=option_rows)
        single_model_options = [dict(row) for row in option_rows if row.get("recommendation_eligible")]
        ensemble_top_models = [dict(row) for row in single_model_options[:5]]
        selection_policy = self._selection_policy(last_result=last_result, option_rows=single_model_options)

        return {
            "resolved_mode": getattr(last_result, "resolved_mode", None) if last_result else None,
            "best_model_id": getattr(last_result, "best_model_id", None) if last_result else None,
            "best_algorithm": getattr(last_result, "best_algorithm", None) if last_result else None,
            "best_promoted_model": best_promoted_model,
            "single_model_options": single_model_options,
            "ensemble_top_models": ensemble_top_models,
            "recommended_models": list((selection_policy or {}).get("selected_models", [])),
            "selection_policy": selection_policy,
            "strategies": sorted(RECOMMEND_STRATEGIES),
            "has_recommendation_models": bool(single_model_options),
            "top_models": single_model_options,
            "model_id_map": {
                row["algorithm"]: row["model_id"]
                for row in single_model_options
                if row.get("model_id") and row.get("algorithm")
            },
        }

    def _resolve_strategy(self, payload: RecommendationPayload) -> str:
        strategy = (payload.strategy or "best_promoted_model").strip().lower()
        if strategy not in RECOMMEND_STRATEGIES:
            raise RecommendationStrategyError(
                "INVALID_STRATEGY",
                f"strategy must be one of {sorted(RECOMMEND_STRATEGIES)}",
                strategy=strategy,
            )
        return strategy

    def _prepare_models(self, payload: RecommendationPayload, strategy: str, last_result, mode: Optional[str]) -> dict:
        if strategy == "best_promoted_model":
            return self._prepare_best_promoted(last_result=last_result)

        if strategy == "single_model":
            request_model = WeightedModelInput(
                model_id=(payload.model_id or "").strip() or None,
                algorithm=(payload.algorithm or "").strip() or None,
                label=(payload.label or "").strip() or None,
                weight=1.0,
                allow_algorithm_fallback=payload.allow_algorithm_fallback or bool(payload.algorithm),
            )
            resolved, warnings = self._resolve_requested_models(
                selected_models=[request_model],
                strategy=strategy,
                mode=mode,
            )
            for model in resolved:
                model["input_weight"] = 1.0
                model["normalized_weight"] = 1.0
            return {
                "models": resolved,
                "weight_normalization_applied": False,
                "weight_normalization_note": "",
                "warnings": warnings,
            }

        selected_models = [
            WeightedModelInput(
                model_id=(selected.model_id or "").strip() or None,
                algorithm=(selected.algorithm or "").strip() or None,
                label=(selected.label or "").strip() or None,
                weight=float(selected.weight),
                allow_algorithm_fallback=selected.allow_algorithm_fallback or bool(selected.algorithm),
            )
            for selected in (payload.models or [])
        ]
        self._validate_ensemble_selection(selected_models)
        resolved, warnings = self._resolve_requested_models(
            selected_models=selected_models,
            strategy=strategy,
            mode=mode,
        )
        resolved, applied, note = self._normalize_weights(resolved, payload.auto_normalize_weights)
        return {
            "models": resolved,
            "weight_normalization_applied": applied,
            "weight_normalization_note": note,
            "warnings": warnings,
        }

    def _prepare_best_promoted(self, last_result=None) -> dict:
        record, warnings = self._resolve_best_promoted_record(last_result=last_result)
        if record is None:
            raise RecommendationStrategyError(
                "NO_PROMOTED_MODEL",
                "No promoted model is available. Train models and promote one first.",
            )

        prepared = self._selected_model_from_record(record=record, input_weight=1.0)
        prepared["normalized_weight"] = 1.0
        return {
            "models": [prepared],
            "weight_normalization_applied": False,
            "weight_normalization_note": "",
            "warnings": warnings,
        }

    def _resolve_best_promoted_record(self, last_result=None) -> tuple[Optional[ModelRecord], list[dict]]:
        warnings: list[dict] = []
        record = None
        best_model_id = getattr(last_result, "best_model_id", None) if last_result else None
        best_algorithm = getattr(last_result, "best_algorithm", None) if last_result else None

        if best_model_id:
            current = self.registry.get(best_model_id)
            if current is None:
                warnings.append(
                    self._warning(
                        "MODEL_ID_NOT_FOUND_FALLBACK_USED",
                        "The last best model_id was not found in the registry; a promoted fallback was used.",
                        requested_model_id=best_model_id,
                    )
                )
            elif current.promoted:
                record = current
            else:
                warnings.append(
                    self._warning(
                        "BEST_MODEL_NOT_PROMOTED_FALLBACK_USED",
                        "The last best model is no longer promoted; a promoted fallback was used.",
                        requested_model_id=best_model_id,
                    )
                )

        if record is None and best_algorithm:
            promoted_for_best_algorithm = self.registry.get_promoted(best_algorithm)
            if promoted_for_best_algorithm:
                record = promoted_for_best_algorithm
                if best_model_id and record.model_id != best_model_id:
                    warnings.append(
                        self._warning(
                            "PROMOTED_POINTER_STALE",
                            "The promoted pointer for the best algorithm changed; the current promoted model was used.",
                            requested_model_id=best_model_id,
                            fallback_model_id=record.model_id,
                            requested_algorithm=best_algorithm,
                        )
                    )

        if record is None:
            record = self._latest_promoted_record()
            if record and best_algorithm and record.algorithm != best_algorithm:
                warnings.append(
                    self._warning(
                        "PROMOTED_ALGORITHM_DIFFERS",
                        "The latest promoted model belongs to a different algorithm than the last best algorithm.",
                        best_algorithm=best_algorithm,
                        promoted_algorithm=record.algorithm,
                        fallback_model_id=record.model_id,
                    )
                )

        return record, warnings

    def _validate_ensemble_selection(self, selected_models: list[WeightedModelInput]) -> None:
        if len(selected_models) < 2:
            raise RecommendationStrategyError(
                "ENSEMBLE_MIN_MODELS",
                "At least 2 models are required for ensemble_weighted.",
            )
        if len(selected_models) > 5:
            raise RecommendationStrategyError(
                "ENSEMBLE_MAX_MODELS",
                "At most 5 models are allowed for ensemble_weighted.",
            )

        seen: set[str] = set()
        for model in selected_models:
            model_id = (model.model_id or "").strip()
            algorithm = (model.algorithm or "").strip()
            if not model_id and not algorithm:
                raise RecommendationStrategyError(
                    "MODEL_SELECTION_REQUIRED",
                    "Each ensemble model must include either model_id or algorithm.",
                )
            dedupe_key = f"model:{model_id}" if model_id else f"algorithm:{algorithm.lower()}"
            if dedupe_key in seen:
                raise RecommendationStrategyError(
                    "DUPLICATE_MODEL_SELECTION",
                    f"Duplicate ensemble selection '{model_id or algorithm}' is not allowed.",
                    duplicate_model_id=model_id,
                    duplicate_algorithm=algorithm or None,
                )
            seen.add(dedupe_key)
            if float(model.weight) <= 0:
                raise RecommendationStrategyError(
                    "INVALID_WEIGHT",
                    f"Model '{model_id or algorithm}' has invalid weight {model.weight}. Weights must be positive.",
                    model_id=model_id or None,
                    algorithm=algorithm or None,
                    weight=model.weight,
                )

    def _resolve_requested_models(
        self,
        selected_models: list[WeightedModelInput],
        strategy: str,
        mode: Optional[str],
    ) -> tuple[list[dict], list[dict]]:
        resolved = []
        warnings = []
        for selected in selected_models:
            record, warning = self._resolve_model_record(selected=selected, strategy=strategy, mode=mode)
            if warning:
                warnings.append(warning)
            resolved.append(
                self._selected_model_from_record(
                    record=record,
                    input_weight=float(selected.weight),
                    label=(selected.label or "").strip() or None,
                )
            )
        return resolved, warnings

    def _resolve_model_record(
        self,
        selected: WeightedModelInput,
        strategy: str,
        mode: Optional[str],
    ) -> tuple[ModelRecord, Optional[dict]]:
        requested_model_id = (selected.model_id or "").strip() or None
        requested_algorithm = (selected.algorithm or "").strip() or None
        allow_fallback = bool(selected.allow_algorithm_fallback)
        warning = None

        if requested_model_id:
            record = self.registry.get(requested_model_id)
            if record is None:
                if allow_fallback and requested_algorithm:
                    record = self.registry.latest_for_algorithm(requested_algorithm)
                    if record is None:
                        raise RecommendationStrategyError(
                            "MODEL_ID_NOT_FOUND",
                            f"Requested model_id '{requested_model_id}' was not found and no fallback model exists for algorithm '{requested_algorithm}'.",
                            requested_model_id=requested_model_id,
                            requested_algorithm=requested_algorithm,
                        )
                    warning = self._warning(
                        "MODEL_ID_NOT_FOUND_FALLBACK_USED",
                        "The requested model_id was not found; the latest compatible model for the requested algorithm was used instead.",
                        requested_model_id=requested_model_id,
                        fallback_model_id=record.model_id,
                        requested_algorithm=requested_algorithm,
                    )
                else:
                    raise RecommendationStrategyError(
                        "MODEL_ID_NOT_FOUND",
                        f"Requested model_id '{requested_model_id}' was not found.",
                        requested_model_id=requested_model_id,
                    )
            if requested_algorithm and record.algorithm != requested_algorithm:
                raise RecommendationStrategyError(
                    "ALGORITHM_VERIFICATION_FAILED",
                    f"Requested model_id '{record.model_id}' belongs to '{record.algorithm}', not '{requested_algorithm}'.",
                    requested_model_id=record.model_id,
                    requested_algorithm=requested_algorithm,
                    resolved_algorithm=record.algorithm,
                )
        else:
            if not allow_fallback or not requested_algorithm:
                raise RecommendationStrategyError(
                    "MODEL_ID_REQUIRED",
                    f"model_id is required when strategy={strategy}.",
                    strategy=strategy,
                )
            record = self.registry.latest_for_algorithm(requested_algorithm)
            if record is None:
                raise RecommendationStrategyError(
                    "NO_MODEL_FOR_ALGORITHM",
                    f"No available model was found for algorithm '{requested_algorithm}'.",
                    requested_algorithm=requested_algorithm,
                )
            warning = self._warning(
                "MODEL_ID_MISSING_FALLBACK_USED",
                "No model_id was provided; the latest compatible model for the requested algorithm was used instead.",
                fallback_model_id=record.model_id,
                requested_algorithm=requested_algorithm,
            )

        if record.algorithm not in ALGORITHM_REGISTRY:
            raise RecommendationStrategyError(
                "UNKNOWN_ALGORITHM",
                f"Unknown algorithm '{record.algorithm}' in model registry.",
                model_id=record.model_id,
                algorithm=record.algorithm,
            )
        if mode and not self._is_mode_compatible(mode, record.algorithm):
            raise RecommendationStrategyError(
                "MODEL_INCOMPATIBLE_WITH_DATASET_MODE",
                f"Algorithm '{record.algorithm}' is incompatible with latest dataset mode '{mode}'.",
                model_id=record.model_id,
                algorithm=record.algorithm,
                resolved_mode=mode,
            )
        return record, warning

    def _run_models(self, selected_models: list[dict], user_id: str, top_n: int) -> tuple[list[dict], list[dict]]:
        outputs = []
        issues = []
        for model in selected_models:
            try:
                loaded = self._load_model(model)
                rec = self.recommender.recommend_with_loaded_model(
                    user_id=user_id,
                    loaded=loaded,
                    top_n=top_n,
                    algorithm_label=model["algorithm"],
                )
            except Exception as e:
                log.warning("Model recommendation failed | algorithm=%s error=%s", model["algorithm"], e)
                issues.append(
                    self._warning(
                        "MODEL_INFERENCE_FAILED",
                        f"Model '{model['algorithm']}' failed during recommendation: {e}",
                        model_id=model["model_id"],
                        algorithm=model["algorithm"],
                    )
                )
                continue

            if not rec.recommendations:
                issues.append(
                    self._warning(
                        "NO_PREDICTIONS_FOR_USER",
                        f"Model '{model['algorithm']}' returned no recommendations for user '{user_id}'.",
                        model_id=model["model_id"],
                        algorithm=model["algorithm"],
                        user_id=user_id,
                    )
                )
                continue

            if rec.algorithm != model["algorithm"]:
                issues.append(
                    self._warning(
                        "USER_NOT_FOUND_POPULARITY_FALLBACK_USED",
                        f"User '{user_id}' was not found for model '{model['algorithm']}'; popularity fallback scores were used.",
                        model_id=model["model_id"],
                        algorithm=model["algorithm"],
                        served_algorithm=rec.algorithm,
                        user_id=user_id,
                    )
                )

            outputs.append(
                {
                    **model,
                    "served_algorithm": rec.algorithm,
                    "recommendations": rec.recommendations,
                }
            )
        return outputs, issues

    def _build_recommendation_option_rows(self, last_result=None) -> list[dict]:
        leaderboard_by_algo = self._leaderboard_rows(last_result)
        training_rows = self._training_option_sources(last_result)
        rows: list[dict] = []
        seen: set[str] = set()

        for source in training_rows:
            algorithm = source.get("algorithm")
            model_id = source.get("model_id")
            record = self.registry.get(model_id) if model_id else None
            if record is None and algorithm:
                record = self.registry.latest_for_algorithm(algorithm)
            row = self._normalize_option_row(
                row=source,
                record=record,
                leaderboard_row=leaderboard_by_algo.get(algorithm),
                default_rank=source.get("rank"),
                last_result=last_result,
            )
            key = row.get("model_id") or f"algorithm:{row.get('algorithm')}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

        if not rows:
            latest_records = self.registry.list_models()[:5]
            for index, record in enumerate(latest_records, start=1):
                row = self._normalize_option_row(
                    row={},
                    record=record,
                    leaderboard_row=leaderboard_by_algo.get(record.algorithm),
                    default_rank=index,
                    last_result=last_result,
                )
                key = row.get("model_id") or f"algorithm:{row.get('algorithm')}"
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)

        rows.sort(key=self._option_sort_key)
        for index, row in enumerate(rows, start=1):
            if row.get("rank") in (None, "", 0):
                row["rank"] = index
        return rows

    def _best_promoted_option(self, last_result=None, option_rows: Optional[list[dict]] = None) -> Optional[dict]:
        option_rows = option_rows or []
        record, warnings = self._resolve_best_promoted_record(last_result=last_result)
        if record is None:
            return {
                "rank": None,
                "model_id": None,
                "algorithm": None,
                "label": "No promoted model available",
                "promoted": False,
                "metric_name": None,
                "metric_value": None,
                "recommendation_eligible": False,
                "feedback_mode": None,
                "status": "unavailable",
                "selection_warnings": warnings,
            }

        for row in option_rows:
            if row.get("model_id") == record.model_id:
                out = dict(row)
                out["selection_warnings"] = warnings
                return out

        leaderboard_by_algo = self._leaderboard_rows(last_result)
        row = self._normalize_option_row(
            row={},
            record=record,
            leaderboard_row=leaderboard_by_algo.get(record.algorithm),
            default_rank=None,
            last_result=last_result,
        )
        row["selection_warnings"] = warnings
        return row

    def _training_option_sources(self, last_result=None) -> list[dict]:
        rows = list(getattr(last_result, "top_model_recommendations", []) or [])
        if rows:
            return [dict(row) for row in rows]

        best_algorithm = getattr(last_result, "best_algorithm", None) if last_result else None
        best_model_id = getattr(last_result, "best_model_id", None) if last_result else None
        if best_algorithm or best_model_id:
            return [
                {
                    "rank": 1,
                    "algorithm": best_algorithm,
                    "model_id": best_model_id,
                    "status": "ok",
                }
            ]
        return []

    def _leaderboard_rows(self, last_result=None) -> dict[str, dict]:
        report = getattr(last_result, "report", None)
        if report is None or not hasattr(report, "leaderboard"):
            return {}
        leaderboard = report.leaderboard()
        if getattr(leaderboard, "empty", True):
            return {}
        return {
            str(row.get("Algorithm")): dict(row)
            for row in leaderboard.to_dict(orient="records")
            if row.get("Algorithm")
        }

    def _normalize_option_row(
        self,
        row: dict,
        record: Optional[ModelRecord],
        leaderboard_row: Optional[dict],
        default_rank: Optional[int],
        last_result=None,
    ) -> dict:
        raw = dict(row or {})
        algorithm = raw.get("algorithm") or getattr(record, "algorithm", None)
        model_id = raw.get("model_id") or getattr(record, "model_id", None)

        metric_name = raw.get("metric_name") or self._metric_name_from_row(leaderboard_row, last_result=last_result)
        metric_value = raw.get("metric_value")
        if metric_value is None and metric_name and leaderboard_row:
            metric_value = leaderboard_row.get(metric_name)
        if metric_value is None and record is not None:
            record_metric_name, record_metric_value = self._metric_from_record(record)
            metric_name = metric_name or record_metric_name
            metric_value = record_metric_value if metric_value is None else metric_value

        promoted = bool(getattr(record, "promoted", False))
        feedback_mode = self._feedback_mode_for_algorithm(algorithm)
        resolved_mode = getattr(last_result, "resolved_mode", None) if last_result else None
        compatible = bool(model_id and algorithm and algorithm in ALGORITHM_REGISTRY)
        if compatible and resolved_mode:
            compatible = self._is_mode_compatible(resolved_mode, algorithm)

        status = raw.get("status") or ("promoted" if promoted else ("saved" if model_id else "unavailable"))
        if not compatible and status not in {"unavailable", "error"}:
            status = "incompatible"

        rank = raw.get("rank")
        if rank is None and leaderboard_row:
            rank = leaderboard_row.get("Rank")
        if rank is None:
            rank = default_rank

        return {
            "rank": self._coerce_int(rank),
            "model_id": model_id,
            "algorithm": algorithm,
            "label": raw.get("label") or self._build_option_label(
                rank=rank,
                algorithm=algorithm,
                metric_name=metric_name,
                metric_value=metric_value,
            ),
            "promoted": promoted,
            "metric_name": metric_name,
            "metric_value": self._coerce_float(metric_value),
            "recommendation_eligible": compatible,
            "feedback_mode": feedback_mode,
            "status": status,
            "summary": raw.get("summary"),
            "performance_summary": raw.get("performance_summary"),
            "fit_summary": raw.get("fit_summary"),
            "reliability_summary": raw.get("reliability_summary"),
            "selection_score_pct": self._coerce_float(raw.get("selection_score_pct")),
            "reasons": list(raw.get("reasons") or []),
            "decision_note": raw.get("decision_note"),
        }

    _normalise_option_row = _normalize_option_row

    def _metric_name_from_row(self, leaderboard_row: Optional[dict], last_result=None) -> Optional[str]:
        report = getattr(last_result, "report", None)
        primary_metric = getattr(report, "primary_metric", None)
        if primary_metric:
            return primary_metric
        if leaderboard_row:
            for key in _METRIC_PRIORITY:
                if key in leaderboard_row:
                    return key
        return None

    def _build_option_label(
        self,
        rank: Optional[int],
        algorithm: Optional[str],
        metric_name: Optional[str],
        metric_value: Optional[float],
    ) -> str:
        prefix = f"#{int(rank)} " if self._coerce_int(rank) else ""
        algo = algorithm or "Unknown model"
        if metric_name and metric_value is not None:
            return f"{prefix}{algo} [{metric_name} {self._format_metric(metric_value)}]"
        return f"{prefix}{algo}"

    def _selected_model_from_record(
        self,
        record: ModelRecord,
        input_weight: float,
        label: Optional[str] = None,
    ) -> dict:
        metric_name, metric_value = self._metric_from_record(record)
        return {
            "model_id": record.model_id,
            "algorithm": record.algorithm,
            "label": label or self._build_option_label(None, record.algorithm, metric_name, metric_value),
            "promoted": bool(record.promoted),
            "metric_name": metric_name,
            "metric_value": metric_value,
            "feedback_mode": self._feedback_mode_for_algorithm(record.algorithm),
            "status": "promoted" if record.promoted else "saved",
            "input_weight": float(input_weight),
            "normalized_weight": float(input_weight),
        }

    def _metric_from_record(self, record: ModelRecord) -> tuple[Optional[str], Optional[float]]:
        metrics = dict(getattr(record, "metrics", {}) or {})
        for key in _METRIC_PRIORITY:
            if key in metrics and metrics.get(key) is not None:
                return key, self._coerce_float(metrics.get(key))
        return None, None

    def _load_model(self, model: dict) -> LoadedModel:
        return self.loader.load_by_id(model["model_id"])

    def _warning(self, _code: str, message: str, **_details) -> str:
        return message

    def _feedback_mode_for_algorithm(self, algorithm: Optional[str]) -> Optional[str]:
        if not algorithm:
            return None
        meta = ALGORITHM_REGISTRY.get(str(algorithm))
        feedback = (meta or {}).get("feedback")
        if feedback in {"explicit", "implicit", "both"}:
            return feedback

        latest = self.registry.latest_for_algorithm(str(algorithm)) if hasattr(self.registry, "latest_for_algorithm") else None
        if latest is not None:
            return "implicit" if bool(getattr(latest, "is_implicit", False)) else "explicit"
        return None

    def _is_mode_compatible(self, mode: Optional[str], algorithm: Optional[str]) -> bool:
        if not algorithm:
            return False
        requested_mode = (mode or "auto").strip().lower()
        feedback_mode = self._feedback_mode_for_algorithm(algorithm)
        allowed = _MODE_TO_ALLOWED_FEEDBACK.get(requested_mode, {"explicit", "implicit", "both"})
        return feedback_mode in allowed

    def _normalize_weights(self, resolved: list[dict], auto_normalize: bool) -> tuple[list[dict], bool, str]:
        if not resolved:
            return [], False, ""

        total = sum(max(float(model.get("input_weight", 0.0)), 0.0) for model in resolved)
        if total <= 0:
            raise RecommendationStrategyError(
                "INVALID_WEIGHT_SUM",
                "Model weights must sum to a positive value.",
            )

        near_one = abs(total - 1.0) < 1e-8
        near_hundred = abs(total - 100.0) < 1e-8
        if not (near_one or near_hundred) and not auto_normalize:
            raise RecommendationStrategyError(
                "WEIGHT_SUM_INVALID",
                "Weights must sum to 1.0 or 100.0 when auto_normalize_weights is disabled.",
                weight_sum=round(total, 6),
            )

        divisor = 1.0 if near_one else (100.0 if near_hundred else total)
        applied = not near_one
        note = ""
        if near_hundred:
            note = "Weights were interpreted as percentages and converted to fractions."
        elif applied:
            note = f"Weights were auto-normalized from total {total:.4f} to sum to 1.0."

        normalized: list[dict] = []
        for model in resolved:
            out = dict(model)
            out["normalized_weight"] = float(model.get("input_weight", 0.0)) / divisor
            normalized.append(out)
        return normalized, applied, note

    def _rebalance_runtime_weights(
        self,
        model_outputs: list[dict],
        had_drops: bool,
    ) -> tuple[list[dict], bool, str]:
        if not model_outputs:
            return [], False, ""

        total = sum(max(float(model.get("normalized_weight", 0.0)), 0.0) for model in model_outputs)
        if not had_drops and abs(total - 1.0) < 1e-8:
            return model_outputs, False, ""

        if total <= 0:
            replacement = 1.0 / len(model_outputs)
            rebalanced = [{**model, "normalized_weight": replacement} for model in model_outputs]
        else:
            rebalanced = [
                {**model, "normalized_weight": float(model.get("normalized_weight", 0.0)) / total}
                for model in model_outputs
            ]
        note = "Weights were rebalanced across the surviving models after runtime drops."
        return rebalanced, True, note

    def _aggregate_weighted(self, model_outputs: list[dict], top_n: int) -> list[dict]:
        item_map: dict[str, dict] = {}
        for model in model_outputs:
            recommendations = list(model.get("recommendations", []) or [])
            numeric_scores = [
                float(row.get("score"))
                for row in recommendations
                if row.get("score") is not None and self._coerce_float(row.get("score")) is not None
            ]
            if not numeric_scores:
                continue
            min_score = min(numeric_scores)
            max_score = max(numeric_scores)
            span = max_score - min_score

            for row in recommendations:
                item_id = row.get("item_id")
                raw_score = self._coerce_float(row.get("score"))
                if item_id is None or raw_score is None:
                    continue
                normalized_score = 1.0 if span == 0 else (raw_score - min_score) / span
                weighted_score = normalized_score * float(model.get("normalized_weight", 0.0))
                bucket = item_map.setdefault(
                    str(item_id),
                    {"item_id": str(item_id), "final_score": 0.0, "contributions": []},
                )
                bucket["final_score"] += weighted_score
                bucket["contributions"].append(
                    {
                        "model_id": model.get("model_id"),
                        "algorithm": model.get("algorithm"),
                        "raw_score": raw_score,
                        "normalized_score": normalized_score,
                        "input_weight": self._coerce_float(model.get("input_weight")),
                        "normalized_weight": self._coerce_float(model.get("normalized_weight")),
                        "weighted_score": weighted_score,
                    }
                )

        ordered = sorted(
            item_map.values(),
            key=lambda row: (-float(row.get("final_score", 0.0)), str(row.get("item_id", ""))),
        )
        recommendations: list[dict] = []
        for index, row in enumerate(ordered[: max(int(top_n), 1)], start=1):
            final_score = float(row.get("final_score", 0.0))
            contributions = sorted(
                row.get("contributions", []),
                key=lambda item: (-float(item.get("weighted_score", 0.0)), str(item.get("algorithm", ""))),
            )
            for contribution in contributions:
                share = 0.0 if final_score <= 0 else (float(contribution["weighted_score"]) / final_score) * 100.0
                contribution["share_pct"] = round(share, 4)
            recommendations.append(
                {
                    "rank": index,
                    "item_id": row["item_id"],
                    "final_score": round(final_score, 6),
                    "contributions": contributions,
                    "explanation": "Weighted ensemble score derived from normalized model contributions.",
                }
            )
        return recommendations

    def _single_model_result(self, model_output: dict, top_n: int) -> list[dict]:
        return self._aggregate_weighted([model_output], top_n=top_n)

    def _build_contribution_breakdown(self, recommendations: list[dict]) -> list[dict]:
        totals: dict[tuple[str, Optional[str]], dict] = {}
        for row in recommendations:
            for contribution in row.get("contributions", []) or []:
                key = (str(contribution.get("algorithm") or ""), contribution.get("model_id"))
                bucket = totals.setdefault(
                    key,
                    {
                        "algorithm": contribution.get("algorithm"),
                        "model_id": contribution.get("model_id"),
                        "weighted_score": 0.0,
                        "item_count": 0,
                    },
                )
                bucket["weighted_score"] += float(contribution.get("weighted_score", 0.0))
                bucket["item_count"] += 1

        total_weighted = sum(row["weighted_score"] for row in totals.values())
        ordered = sorted(
            totals.values(),
            key=lambda row: (-float(row["weighted_score"]), str(row.get("algorithm", ""))),
        )
        for row in ordered:
            share = 0.0 if total_weighted <= 0 else (row["weighted_score"] / total_weighted) * 100.0
            row["share_pct"] = round(share, 4)
            row["weighted_score"] = round(float(row["weighted_score"]), 6)
        return ordered

    def _latest_promoted_record(self) -> Optional[ModelRecord]:
        promoted = self.registry.list_models(promoted_only=True)
        return promoted[0] if promoted else None

    def _selection_policy(self, last_result=None, option_rows: Optional[list[dict]] = None) -> Optional[dict]:
        if last_result is not None:
            existing = getattr(last_result, "model_selection_policy", None)
            if existing:
                return existing

        rows = [dict(row) for row in (option_rows or []) if row.get("algorithm")]
        if not rows:
            return None

        selected = rows[:1]
        return {
            "selection_type": "single_model",
            "selected_count": 1,
            "selected_models": selected,
            "reason": f"{selected[0].get('algorithm')} is the highest-ranked available model.",
            "display_title": "Recommended single model",
            "recommended_strategy": "single_model",
        }

    def _option_sort_key(self, row: dict) -> tuple:
        rank = self._coerce_int(row.get("rank"))
        metric_name = row.get("metric_name")
        metric_value = self._coerce_float(row.get("metric_value"))
        if metric_name == "RMSE":
            metric_key = metric_value if metric_value is not None else float("inf")
        else:
            metric_key = -(metric_value if metric_value is not None else float("-inf"))
        return (
            0 if row.get("recommendation_eligible") else 1,
            0 if row.get("promoted") else 1,
            rank if rank is not None else 10**9,
            metric_key,
            str(row.get("algorithm") or ""),
        )

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        try:
            if value in (None, ""):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            if value in (None, ""):
                return None
            out = float(value)
        except (TypeError, ValueError):
            return None
        if out != out or out in {float("inf"), float("-inf")}:
            return None
        return out

    @staticmethod
    def _format_metric(value: float) -> str:
        return f"{float(value):.4f}"
