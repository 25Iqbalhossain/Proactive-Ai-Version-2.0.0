"""
End-to-end training pipeline.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from colorama import Fore, Style

from algorithms import ALGORITHM_REGISTRY
from benchmark.benchmark_engine import BenchmarkEngine, BenchmarkReport
from config.database import NoSQLConfig, SQLConfig
from config.settings import (
    ALGORITHM_MODES,
    DEFAULT_ALGORITHM_MODE,
    OPTUNA_TRIALS_AUTO,
    TOP_K,
    TOP_K_ALLOWED,
    TOP_MODEL_ALLOWED,
    TOP_N_MODELS,
)
from data_processing.data_cleaning import DataCleaner
from data_processing.dataset_analyzer import DatasetAnalyzer
from data_processing.feedback_detector import FeedbackDetector
from data_processing.interaction_matrix import InteractionMatrixBuilder
from ingestion.csv_loader import FileLoader
from ingestion.nosql_connector import NoSQLConnector
from ingestion.sql_query_executor import SQLQueryExecutor
from models.model_registry import ModelRegistry
from optimization.optuna_tuner import OptunaTuner, TuningResult, get_optuna_trial_budgets
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TrainingConfig:
    top_k: int = TOP_K
    n_tuning_trials: int = OPTUNA_TRIALS_AUTO
    top_model_count: int = TOP_N_MODELS
    algorithm_mode: str = DEFAULT_ALGORITHM_MODE
    force_all_algos: bool = False
    interactive: bool = False
    save_model: bool = True
    auto_promote: bool = True

    def validate(self) -> None:
        if self.top_k not in TOP_K_ALLOWED:
            raise ValueError(f"top_k must be one of {TOP_K_ALLOWED}, got {self.top_k}")
        if self.top_model_count not in TOP_MODEL_ALLOWED:
            raise ValueError(
                f"top_model_count must be one of {TOP_MODEL_ALLOWED}, got {self.top_model_count}"
            )
        if self.algorithm_mode not in ALGORITHM_MODES:
            raise ValueError(
                f"algorithm_mode must be one of {ALGORITHM_MODES}, got {self.algorithm_mode}"
            )


@dataclass
class TrainingResult:
    report: BenchmarkReport
    tuning_results: list[TuningResult]
    best_model_id: Optional[str]
    best_algorithm: Optional[str]
    best_params: dict
    elapsed_s: float
    all_model_ids: list[str] = field(default_factory=list)
    resolved_mode: str = "implicit"
    feedback_profile: dict = field(default_factory=dict)
    top_model_recommendations: list[dict] = field(default_factory=list)
    ranking_logic: dict = field(default_factory=dict)
    optuna_note: str = ""
    optuna_policy: dict = field(default_factory=dict)
    model_selection_policy: dict = field(default_factory=dict)

    @property
    def all_tuning_results(self) -> list[TuningResult]:
        return self.tuning_results


class TrainingPipeline:
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.config.validate()

        self.analyzer = DatasetAnalyzer()
        self.detector = FeedbackDetector()
        self.cleaner = DataCleaner()
        self.builder = InteractionMatrixBuilder()
        self.benchmark = BenchmarkEngine()
        self.tuner = OptunaTuner()
        self.registry = ModelRegistry()

    def run_from_file(self, path: str, file_format: str = "auto") -> TrainingResult:
        print(f"\n{Fore.CYAN}Loading from file: {path}{Style.RESET_ALL}")
        loader = FileLoader()
        df = loader.load(path, file_format=file_format)
        return self._run(df)

    def run_from_dataframe(self, df: pd.DataFrame) -> TrainingResult:
        print(f"\n{Fore.CYAN}Running from in-memory DataFrame{Style.RESET_ALL}")
        return self._run(df)

    def run_from_sql(
        self,
        sql_config: SQLConfig,
        sql: str,
        params: Optional[dict] = None,
    ) -> TrainingResult:
        print(f"\n{Fore.CYAN}Connecting to {sql_config.dialect} database...{Style.RESET_ALL}")
        executor = SQLQueryExecutor(sql_config)
        query_result = executor.execute_custom(sql, params)
        return self._run(query_result.df)

    def run_from_nosql(
        self,
        nosql_config: NoSQLConfig,
        collection: str,
        query: Optional[dict] = None,
    ) -> TrainingResult:
        print(f"\n{Fore.CYAN}Connecting to {nosql_config.engine}...{Style.RESET_ALL}")
        with NoSQLConnector(nosql_config) as conn:
            if nosql_config.engine == "mongodb":
                df = conn.fetch_collection(collection, query)
            elif nosql_config.engine == "dynamodb":
                df = conn.fetch_dynamodb_table(collection)
            else:
                raise ValueError(f"Unsupported NoSQL engine: {nosql_config.engine}")
        return self._run(df)

    def _run(self, raw_df: pd.DataFrame) -> TrainingResult:
        t0 = time.time()
        self._print_header()
        print(f"Rows: {raw_df.shape[0]:,} | Columns: {raw_df.shape[1]}")

        print(f"\n{Fore.YELLOW}[1/6] Column Detection{Style.RESET_ALL}")
        mapping = self.analyzer.detect_columns(raw_df)
        warnings = self.analyzer.validate_mapping(raw_df, mapping)
        for warning in warnings:
            print(f"{Fore.YELLOW}warning: {warning}{Style.RESET_ALL}")
        mapping = self.analyzer.confirm_or_override(
            mapping, list(raw_df.columns), interactive=self.config.interactive
        )
        if not mapping.userID or not mapping.itemID:
            raise ValueError(
                "Dataset validation failed: could not confidently detect both userID and itemID columns. "
                f"Detected mapping was userID={mapping.userID!r}, itemID={mapping.itemID!r}, "
                f"rating={mapping.rating!r}, timestamp={mapping.timestamp!r}. "
                "Inspect the built CSV/JSON and ensure it contains real user-item interaction identifiers."
            )

        print(f"\n{Fore.YELLOW}[2/6] Feedback Profiling{Style.RESET_ALL}")
        probe_df = raw_df.rename(columns={mapping.rating: "rating"}) if mapping.rating else raw_df
        feedback = self.detector.detect_from_df(probe_df)
        resolved_mode = self.detector.resolve_mode(feedback, self.config.algorithm_mode)
        print(f"Resolved mode: {resolved_mode.upper()} (requested={self.config.algorithm_mode})")

        print(f"\n{Fore.YELLOW}[3/6] Data Cleaning{Style.RESET_ALL}")
        clean_df, cleaning_report = self.cleaner.clean(raw_df, mapping)
        if clean_df.empty:
            raise ValueError(
                "Dataset validation failed after cleaning: 0 interactions remained. "
                f"Detected mapping was userID={mapping.userID!r}, itemID={mapping.itemID!r}, "
                f"rating={mapping.rating!r}, timestamp={mapping.timestamp!r}. "
                f"Available columns: {list(raw_df.columns)}"
            )
        # For hybrid mode we optimize ranking metrics in Optuna.
        clean_df.attrs["is_implicit"] = resolved_mode in {"implicit", "hybrid"}
        clean_df.attrs["resolved_mode"] = resolved_mode
        train, test = self.cleaner.split(clean_df)

        print(f"\n{Fore.YELLOW}[4/6] Interaction Matrix{Style.RESET_ALL}")
        im = self.builder.build(train)
        print(f"Matrix: {im.n_users:,} x {im.n_items:,} nnz={im.matrix.nnz:,} density={im.density:.4%}")

        print(f"\n{Fore.YELLOW}[5/6] Benchmark{Style.RESET_ALL}")
        report = self.benchmark.run(
            train=train,
            test=test,
            top_k=self.config.top_k,
            force_all=self.config.force_all_algos,
            algorithm_mode=resolved_mode,
            feedback_profile=feedback,
        )
        self._print_leaderboard(report)

        print(f"\n{Fore.YELLOW}[6/6] Hyperparameter Tuning{Style.RESET_ALL}")
        top_n = self.config.top_model_count
        top_algorithms = [r.algorithm for r in report.top_n(top_n)]
        print(f"Selected algorithms ({len(top_algorithms)}): {top_algorithms}")

        bench_by_algo = {r.algorithm: r for r in report.results}
        tuning_results: list[TuningResult] = []
        optuna_policy = self._build_optuna_policy(
            selected_algorithms=top_algorithms,
            primary_metric=report.primary_metric,
            primary_metric_direction=report.primary_metric_direction,
        )
        optuna_note = optuna_policy.get("summary", "")
        if self.config.n_tuning_trials == 0:
            print("Optuna skipped (n_tuning_trials=0).")
            for algo in top_algorithms:
                tuning_results.append(
                    TuningResult(
                        algorithm=algo,
                        best_params={},
                        best_value=self._benchmark_primary_value(report, bench_by_algo.get(algo)),
                        metric_name=report.primary_metric,
                        n_trials=0,
                        elapsed_s=0.0,
                        status="skipped",
                        fallback_reason="tuning skipped by configuration",
                        trial_budget="fixed(0)",
                    )
                )
        else:
            tuning_results = self.tuner.tune_top_n(
                algorithms=top_algorithms,
                train=report.train,
                test=report.test,
                n_trials=self.config.n_tuning_trials,
                top_k=self.config.top_k,
            )

        ranked_tuning = self._rank_tuning_results(tuning_results, report, bench_by_algo)
        best_model_id = None
        best_algorithm = None
        best_params: dict = {}
        all_model_ids: list[str] = []

        if self.config.save_model:
            for rank, tune_result in enumerate(ranked_tuning, 1):
                promote = rank == 1 and self.config.auto_promote
                try:
                    model_id = self._save_model(
                        tune_result=tune_result,
                        report=report,
                        rank=rank,
                        total=len(ranked_tuning),
                        promote=promote,
                    )
                    all_model_ids.append(model_id)
                    if rank == 1:
                        best_model_id = model_id
                        best_algorithm = tune_result.algorithm
                        best_params = tune_result.best_params
                except Exception as e:
                    log.error("Model save failed | algo=%s error=%s", tune_result.algorithm, e, exc_info=True)
        elif ranked_tuning:
            best_algorithm = ranked_tuning[0].algorithm
            best_params = ranked_tuning[0].best_params

        recommendations = self._build_model_recommendations(
            ranked_results=ranked_tuning,
            report=report,
            bench_by_algo=bench_by_algo,
            model_ids=all_model_ids,
            feedback_profile=feedback.to_dict(),
            cleaning_report=cleaning_report.__dict__,
        )
        selection_policy = self._build_model_selection_policy(
            recommendations=recommendations,
            primary_metric=report.primary_metric,
            primary_metric_direction=report.primary_metric_direction,
        )

        elapsed = time.time() - t0
        self._print_footer(best_algorithm, best_params, all_model_ids, elapsed)
        return TrainingResult(
            report=report,
            tuning_results=ranked_tuning,
            best_model_id=best_model_id,
            best_algorithm=best_algorithm,
            best_params=best_params,
            elapsed_s=elapsed,
            all_model_ids=all_model_ids,
            resolved_mode=resolved_mode,
            feedback_profile=feedback.to_dict(),
            top_model_recommendations=recommendations,
            ranking_logic=report.ranking_logic,
            optuna_note=optuna_note,
            optuna_policy=optuna_policy,
            model_selection_policy=selection_policy,
        )

    def _rank_tuning_results(
        self,
        tuning_results: list[TuningResult],
        report: BenchmarkReport,
        bench_by_algo: dict[str, object],
    ) -> list[TuningResult]:
        direction = report.primary_metric_direction

        def score(result: TuningResult) -> float:
            if result.best_value is not None and math.isfinite(float(result.best_value)):
                return float(result.best_value)
            benchmark_result = bench_by_algo.get(result.algorithm)
            if benchmark_result:
                return self._benchmark_primary_value(report, benchmark_result)
            return float("-inf") if direction == "maximize" else float("inf")

        def key(result: TuningResult) -> tuple[int, float]:
            status_rank = {"ok": 0, "fallback": 1, "skipped": 2, "failed": 3}.get(result.status, 4)
            raw = score(result)
            metric_key = -raw if direction == "maximize" else raw
            return status_rank, metric_key

        out = list(tuning_results)
        out.sort(key=key)
        return out

    @staticmethod
    def _benchmark_primary_value(report: BenchmarkReport, bench_result) -> float:
        if not bench_result:
            return 0.0
        val = bench_result.metrics.get(report.primary_metric)
        if val is None:
            return 0.0
        try:
            return float(val)
        except Exception:
            return 0.0

    def _save_model(
        self,
        tune_result: TuningResult,
        report: BenchmarkReport,
        rank: int,
        total: int,
        promote: bool,
    ) -> str:
        algo = tune_result.algorithm
        meta = ALGORITHM_REGISTRY[algo]
        params = tune_result.best_params or {}
        print(f"  [{rank}/{total}] retraining {algo} with params={params}")

        run_out = meta["fn"](
            report.train,
            report.test,
            self.config.top_k,
            **params,
            return_model=True,
        )
        if len(run_out) != 3:
            raise RuntimeError(
                f"Algorithm '{algo}' did not return a trained model object with return_model=True"
            )

        _, _, model_obj = run_out
        metrics = dict(next((r.metrics for r in report.results if r.algorithm == algo), {}))
        if tune_result.best_value is not None:
            metrics[tune_result.metric_name] = tune_result.best_value
        metrics["Composite Score"] = next(
            (
                rec.get("Composite Score")
                for rec in report.leaderboard().to_dict(orient="records")
                if rec.get("Algorithm") == algo
            ),
            None,
        )
        model_id = self.registry.save(
            model=model_obj,
            algorithm=algo,
            metrics=metrics,
            params=params,
            is_implicit=report.resolved_mode in {"implicit", "hybrid"},
            notes=(
                f"rank={rank}/{total} status={tune_result.status} "
                f"trials={tune_result.n_trials} metric={tune_result.metric_name}"
            ),
        )
        if promote:
            self.registry.promote(model_id)
        return model_id

    def _build_model_recommendations(
        self,
        ranked_results: list[TuningResult],
        report: BenchmarkReport,
        bench_by_algo: dict[str, object],
        model_ids: list[str],
        feedback_profile: dict,
        cleaning_report: dict,
    ) -> list[dict]:
        lb = report.leaderboard().set_index("Algorithm") if not report.leaderboard().empty else pd.DataFrame()
        sparsity = cleaning_report.get("sparsity", 1.0)
        sparse_label = "sparse" if sparsity >= 0.98 else "dense"
        out: list[dict] = []
        metric_values = []
        composite_values = []
        for tune in ranked_results:
            algo = tune.algorithm
            row = lb.loc[algo] if not lb.empty and algo in lb.index else None
            perf_value = (
                tune.best_value
                if tune.best_value is not None
                else (float(row.get(report.primary_metric)) if row is not None else None)
            )
            composite_value = float(row.get("Composite Score")) if row is not None else None
            if isinstance(perf_value, (int, float)) and math.isfinite(float(perf_value)):
                metric_values.append(float(perf_value))
            if isinstance(composite_value, (int, float)) and math.isfinite(float(composite_value)):
                composite_values.append(float(composite_value))

        metric_min = min(metric_values) if metric_values else None
        metric_max = max(metric_values) if metric_values else None
        composite_min = min(composite_values) if composite_values else None
        composite_max = max(composite_values) if composite_values else None

        for idx, tune in enumerate(ranked_results, 1):
            algo = tune.algorithm
            meta = ALGORITHM_REGISTRY.get(algo, {})
            row = lb.loc[algo] if not lb.empty and algo in lb.index else None
            perf_value = (
                tune.best_value
                if tune.best_value is not None
                else (float(row.get(report.primary_metric)) if row is not None else None)
            )
            elapsed_s = float(row.get("Time (s)")) if row is not None and pd.notna(row.get("Time (s)")) else None
            composite_score = float(row.get("Composite Score")) if row is not None else None
            selection_score_pct = self._estimate_selection_score_pct(
                metric_name=report.primary_metric,
                metric_value=perf_value,
                metric_direction=report.primary_metric_direction,
                composite_score=composite_score,
                metric_min=metric_min,
                metric_max=metric_max,
                composite_min=composite_min,
                composite_max=composite_max,
            )
            performance_summary = (
                f"{report.primary_metric} reached {perf_value:.4f}."
                if isinstance(perf_value, (int, float))
                else f"{report.primary_metric} was not available after tuning."
            )
            fit_summary = (
                f"{algo} matches the {report.resolved_mode} dataset mode and is rated {meta.get('sparsity_fit', 'medium')} for {sparse_label} data."
            )
            reliability_summary = (
                f"Tuning finished with status '{tune.status}' and trial budget {tune.trial_budget or 'n/a'}."
            )
            reasons = [
                f"Feedback fit: algorithm supports {meta.get('feedback', 'unknown')} mode; dataset resolved as {report.resolved_mode}.",
                f"Expected performance: {report.primary_metric}={perf_value:.4f}."
                if isinstance(perf_value, (int, float))
                else f"Expected performance: benchmark primary metric is {report.primary_metric}.",
                f"Scalability: {meta.get('scalability', 'medium')} with limits users={meta.get('max_users') or 'unbounded'}, items={meta.get('max_items') or 'unbounded'}.",
                f"Training speed: {meta.get('training_speed', 'medium')}"
                + (f" (observed {elapsed_s:.2f}s)." if elapsed_s is not None else "."),
                f"Robustness: {meta.get('robustness', 'medium')} (tuning status={tune.status}).",
                f"Data suitability: rated for {meta.get('sparsity_fit', 'medium')} sparsity; current dataset is {sparse_label} ({sparsity:.2%} sparsity).",
                f"Interpretability: {meta.get('interpretability', 'medium')}.",
                f"Production readiness: {meta.get('production_readiness', 'medium')}.",
            ]
            out.append(
                {
                    "rank": idx,
                    "algorithm": algo,
                    "model_id": model_ids[idx - 1] if idx - 1 < len(model_ids) else None,
                    "metric_name": report.primary_metric,
                    "metric_value": perf_value,
                    "composite_score": composite_score,
                    "status": tune.status,
                    "trial_budget": tune.trial_budget,
                    "reasons": reasons,
                    "summary": " ".join([performance_summary, fit_summary, reliability_summary]),
                    "performance_summary": performance_summary,
                    "fit_summary": fit_summary,
                    "reliability_summary": reliability_summary,
                    "selection_score_pct": selection_score_pct,
                    "feedback_profile": feedback_profile,
                }
            )
        self._annotate_ranked_recommendations(
            ranked=out,
            primary_metric=report.primary_metric,
            primary_metric_direction=report.primary_metric_direction,
        )
        return out

    def _build_optuna_policy(
        self,
        selected_algorithms: list[str],
        primary_metric: str,
        primary_metric_direction: str,
    ) -> dict:
        requested_trials = int(self.config.n_tuning_trials)
        adaptive_budgets = get_optuna_trial_budgets()
        selected_budgets = {
            algorithm: adaptive_budgets.get(algorithm)
            for algorithm in selected_algorithms
            if adaptive_budgets.get(algorithm) is not None
        }
        if requested_trials == OPTUNA_TRIALS_AUTO:
            mode = "adaptive"
            summary = (
                "Optuna uses adaptive per-algorithm trial budgets because n_trials=-1. "
                f"Each selected algorithm receives its own preset budget before ranking on {primary_metric}."
            )
        elif requested_trials == 0:
            mode = "disabled"
            summary = (
                "Optuna is disabled because n_trials=0. "
                "The system keeps benchmark scores and skips hyperparameter search."
            )
        else:
            mode = "fixed"
            summary = (
                f"Optuna uses a fixed budget of {requested_trials} trials for every selected algorithm."
            )

        return {
            "requested_trials": requested_trials,
            "mode": mode,
            "top_k": int(self.config.top_k),
            "top_k_definition": (
                f"Top-K={self.config.top_k} means ranking metrics evaluate whether relevant items appear "
                f"in each user's top {self.config.top_k} recommendations."
            ),
            "objective_metric": primary_metric,
            "objective_direction": primary_metric_direction,
            "selected_algorithms": list(selected_algorithms),
            "adaptive_trial_budgets": selected_budgets if requested_trials == OPTUNA_TRIALS_AUTO else {},
            "rules": [
                "n_trials=-1 uses adaptive trial budgets per algorithm.",
                "n_trials=0 skips Optuna and keeps benchmark scores.",
                "n_trials>0 applies the same fixed trial count to every selected algorithm.",
                f"Implicit and hybrid datasets optimize {primary_metric} by ranking quality; explicit datasets minimize RMSE.",
            ],
            "summary": summary,
        }

    def _build_model_selection_policy(
        self,
        recommendations: list[dict],
        primary_metric: str,
        primary_metric_direction: str,
    ) -> dict:
        ranked = [dict(row) for row in recommendations if row.get("algorithm")]
        if not ranked:
            return {
                "selection_type": "none",
                "selected_count": 0,
                "selected_models": [],
                "reason": "No ranked models were available after training.",
                "display_title": "No recommended models",
                "recommended_strategy": None,
            }

        self._annotate_ranked_recommendations(
            ranked=ranked,
            primary_metric=primary_metric,
            primary_metric_direction=primary_metric_direction,
        )

        top = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        top_score = float(top.get("selection_score_pct") or 0.0)
        runner_up_score = float(runner_up.get("selection_score_pct") or 0.0) if runner_up else 0.0
        score_gap = round(top_score - runner_up_score, 2) if runner_up else 100.0
        shortlist_count = max(1, min(len(ranked), int(self.config.top_model_count)))
        display_selected = ranked[:shortlist_count]

        if len(ranked) == 1 or top_score >= 90.0 or (top_score >= 75.0 and score_gap >= 10.0):
            serving_selected = ranked[:1]
            serving_reason = (
                f"{top['algorithm']} is the clear winner with a selection score of {top_score:.2f}%."
                if runner_up is None
                else (
                    f"{top['algorithm']} is clearly ahead with a selection score of {top_score:.2f}% "
                    f"and a {score_gap:.2f}-point lead over {runner_up['algorithm']}. "
                    "The API will return one recommended model by default."
                )
            )
        else:
            serving_selected = ranked[: min(shortlist_count, 5)]
            serving_reason = (
                f"No single model separated enough from the pack. {top['algorithm']} leads with "
                f"{top_score:.2f}% selection score, but the gap to {runner_up['algorithm']} is only "
                f"{score_gap:.2f} points. The API will return the best {len(serving_selected)} models for comparison."
                if runner_up is not None
                else f"{top['algorithm']} is currently the only available model."
            )

        selection_type = "single_model" if len(display_selected) == 1 else "top_ranked_models"
        display_title = (
            "Recommended single model"
            if len(display_selected) == 1
            else f"Top {len(display_selected)} model recommendations and reasons"
        )
        recommended_strategy = "single_model" if len(serving_selected) == 1 else "ensemble_weighted"
        reason = (
            f"{serving_reason} Showing the top {len(display_selected)} ranked models so you can compare why each one scored well."
            if len(display_selected) > 1
            else serving_reason
        )
        best_model_explanation = top.get("reason") or serving_reason
        comparison_explanations = [
            row["comparison_to_next"]
            for row in display_selected
            if row.get("comparison_to_next")
        ]

        selected_algorithms = {row.get("algorithm") for row in display_selected}
        for row in ranked:
            row["selected_by_policy"] = row.get("algorithm") in selected_algorithms
            if row["selected_by_policy"]:
                row["decision_note"] = reason

        return {
            "selection_type": selection_type,
            "selected_count": len(display_selected),
            "selected_models": display_selected,
            "reason": reason,
            "display_title": display_title,
            "recommended_strategy": recommended_strategy,
            "serving_selected_count": len(serving_selected),
            "serving_selected_models": serving_selected,
            "serving_reason": serving_reason,
            "best_model_explanation": best_model_explanation,
            "comparison_explanations": comparison_explanations,
            "primary_metric": primary_metric,
            "primary_metric_direction": primary_metric_direction,
            "top_model": {
                "algorithm": top.get("algorithm"),
                "model_id": top.get("model_id"),
                "selection_score_pct": top.get("selection_score_pct"),
                "metric_name": top.get("metric_name"),
                "metric_value": top.get("metric_value"),
            },
            "runner_up": (
                {
                    "algorithm": runner_up.get("algorithm"),
                    "model_id": runner_up.get("model_id"),
                    "selection_score_pct": runner_up.get("selection_score_pct"),
                    "metric_name": runner_up.get("metric_name"),
                    "metric_value": runner_up.get("metric_value"),
                }
                if runner_up
                else None
            ),
            "score_gap_pct": score_gap,
            "decision_rules": [
                "Always show the full requested top-model count in the training results so the user can compare ranked candidates.",
                "Use one serving model when the winner is dominant enough to stand on its own.",
                f"Use up to {min(shortlist_count, 5)} serving models when several candidates remain competitive because the ensemble API accepts at most 5 models.",
                "Dominance is estimated from the primary metric, composite score, and relative separation between models.",
            ],
        }

    def _annotate_ranked_recommendations(
        self,
        ranked: list[dict],
        primary_metric: str,
        primary_metric_direction: str,
    ) -> None:
        ordered = sorted(
            ranked,
            key=lambda row: (
                int(row.get("rank") or 10**9),
                -float(row.get("selection_score_pct") or 0.0),
                str(row.get("algorithm") or ""),
            ),
        )
        for index, row in enumerate(ordered):
            previous_row = ordered[index - 1] if index > 0 else None
            next_row = ordered[index + 1] if index + 1 < len(ordered) else None
            row["comparison_to_previous"] = (
                self._build_rank_comparison(
                    higher=previous_row,
                    lower=row,
                    primary_metric=primary_metric,
                    primary_metric_direction=primary_metric_direction,
                )
                if previous_row
                else None
            )
            row["comparison_to_next"] = (
                self._build_rank_comparison(
                    higher=row,
                    lower=next_row,
                    primary_metric=primary_metric,
                    primary_metric_direction=primary_metric_direction,
                )
                if next_row
                else None
            )
            row["reason"] = self._build_rank_reason(
                row=row,
                previous_row=previous_row,
                next_row=next_row,
                primary_metric=primary_metric,
                primary_metric_direction=primary_metric_direction,
            )

    def _build_rank_reason(
        self,
        row: dict,
        previous_row: Optional[dict],
        next_row: Optional[dict],
        primary_metric: str,
        primary_metric_direction: str,
    ) -> str:
        rank = int(row.get("rank") or 0)
        algorithm = row.get("algorithm") or "Unknown model"
        score = self._safe_float(row.get("selection_score_pct"))
        parts = []
        if rank == 1:
            if score is not None:
                parts.append(f"Ranked #{rank} because it has the highest selection score at {score:.2f}%.")
            else:
                parts.append(f"Ranked #{rank} because it is the highest-scoring supported model.")
        else:
            if score is not None:
                parts.append(f"Ranked #{rank} with a selection score of {score:.2f}%.")
            else:
                parts.append(f"Ranked #{rank} in the supported model list.")

        if next_row is not None:
            parts.append(
                self._build_rank_comparison(
                    higher=row,
                    lower=next_row,
                    primary_metric=primary_metric,
                    primary_metric_direction=primary_metric_direction,
                )
            )
        elif previous_row is not None:
            parts.append(
                f"It remains behind #{int(previous_row.get('rank') or rank - 1)} {previous_row.get('algorithm')} and has no lower-ranked supported competitor below it."
            )

        if row.get("performance_summary"):
            parts.append(str(row["performance_summary"]))
        if row.get("fit_summary"):
            parts.append(str(row["fit_summary"]))
        if row.get("reliability_summary"):
            parts.append(str(row["reliability_summary"]))
        return " ".join(part for part in parts if part)

    def _build_rank_comparison(
        self,
        higher: Optional[dict],
        lower: Optional[dict],
        primary_metric: str,
        primary_metric_direction: str,
    ) -> str:
        if not higher or not lower:
            return ""
        higher_name = higher.get("algorithm") or "Higher-ranked model"
        lower_name = lower.get("algorithm") or "Lower-ranked model"
        higher_rank = int(higher.get("rank") or 0)
        lower_rank = int(lower.get("rank") or 0)

        higher_score = self._safe_float(higher.get("selection_score_pct"))
        lower_score = self._safe_float(lower.get("selection_score_pct"))
        score_text = ""
        if higher_score is not None and lower_score is not None:
            gap = higher_score - lower_score
            score_text = (
                f"{higher_name} stays ahead of #{lower_rank} {lower_name} because its selection score is "
                f"{higher_score:.2f}% versus {lower_score:.2f}% ({gap:+.2f} points)."
            )
        else:
            score_text = f"{higher_name} stays ahead of #{lower_rank} {lower_name} on the ranking composite."

        detail_parts = []
        metric_sentence = self._metric_comparison_sentence(
            higher=higher,
            lower=lower,
            primary_metric=primary_metric,
            primary_metric_direction=primary_metric_direction,
        )
        if metric_sentence:
            detail_parts.append(metric_sentence)

        higher_composite = self._safe_float(higher.get("composite_score"))
        lower_composite = self._safe_float(lower.get("composite_score"))
        if higher_composite is not None and lower_composite is not None:
            detail_parts.append(
                f"Composite Score is also stronger ({higher_composite:.4f} vs {lower_composite:.4f})."
            )
        return " ".join([score_text, *detail_parts]).strip()

    def _metric_comparison_sentence(
        self,
        higher: dict,
        lower: dict,
        primary_metric: str,
        primary_metric_direction: str,
    ) -> str:
        higher_metric = self._safe_float(higher.get("metric_value"))
        lower_metric = self._safe_float(lower.get("metric_value"))
        if higher_metric is None or lower_metric is None:
            return ""
        direction = (primary_metric_direction or "").strip().lower()
        if direction.startswith("min"):
            better_text = f"lower {primary_metric}"
            higher_is_better = higher_metric <= lower_metric
        else:
            better_text = f"higher {primary_metric}"
            higher_is_better = higher_metric >= lower_metric
        if not higher_is_better:
            return ""
        return f"It also posts {better_text} ({higher_metric:.4f} vs {lower_metric:.4f})."

    @staticmethod
    def _safe_float(value: object) -> Optional[float]:
        try:
            if value in (None, ""):
                return None
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out):
            return None
        return out

    @staticmethod
    def _estimate_selection_score_pct(
        metric_name: str,
        metric_value: Optional[float],
        metric_direction: str,
        composite_score: Optional[float],
        metric_min: Optional[float],
        metric_max: Optional[float],
        composite_min: Optional[float],
        composite_max: Optional[float],
    ) -> float:
        signals: list[tuple[float, float]] = []

        bounded_metric = None
        if isinstance(metric_value, (int, float)) and math.isfinite(float(metric_value)):
            metric_value = float(metric_value)
            if metric_direction == "maximize" and 0.0 <= metric_value <= 1.0:
                bounded_metric = metric_value * 100.0
            elif str(metric_name).upper() == "RMSE":
                bounded_metric = 100.0 / (1.0 + max(metric_value, 0.0))
        if bounded_metric is not None:
            signals.append((bounded_metric, 0.5))

        if isinstance(composite_score, (int, float)) and math.isfinite(float(composite_score)):
            composite_score = float(composite_score)
            if 0.0 <= composite_score <= 1.0:
                signals.append((composite_score * 100.0, 0.25))

        normalized_metric = None
        if (
            isinstance(metric_value, (int, float))
            and metric_min is not None
            and metric_max is not None
            and math.isfinite(float(metric_min))
            and math.isfinite(float(metric_max))
        ):
            span = float(metric_max) - float(metric_min)
            if span <= 1e-12:
                normalized_metric = 100.0
            elif metric_direction == "minimize":
                normalized_metric = ((float(metric_max) - float(metric_value)) / span) * 100.0
            else:
                normalized_metric = ((float(metric_value) - float(metric_min)) / span) * 100.0
        if normalized_metric is not None:
            signals.append((max(0.0, min(100.0, normalized_metric)), 0.25))

        if (
            isinstance(composite_score, (int, float))
            and composite_min is not None
            and composite_max is not None
            and math.isfinite(float(composite_min))
            and math.isfinite(float(composite_max))
        ):
            span = float(composite_max) - float(composite_min)
            if span <= 1e-12:
                normalized_composite = 100.0
            else:
                normalized_composite = ((float(composite_score) - float(composite_min)) / span) * 100.0
            signals.append((max(0.0, min(100.0, normalized_composite)), 0.25))

        if not signals:
            return 0.0

        weighted_total = sum(value * weight for value, weight in signals)
        total_weight = sum(weight for _, weight in signals)
        return round(weighted_total / total_weight, 2)

    @staticmethod
    def _print_header() -> None:
        print(f"\n{Fore.CYAN}{'=' * 72}")
        print("  PROACTIVE AI - TRAINING PIPELINE")
        print(f"{'=' * 72}{Style.RESET_ALL}")

    @staticmethod
    def _print_footer(algo, params, all_model_ids, elapsed):
        print(f"\n{Fore.CYAN}{'=' * 72}")
        print(f"TRAINING COMPLETE in {elapsed:.1f}s")
        if algo:
            print(f"Best algorithm: {algo}")
            print(f"Best params   : {params}")
        print(f"Models saved  : {len(all_model_ids)}")
        print(f"{'=' * 72}{Style.RESET_ALL}\n")

    @staticmethod
    def _print_leaderboard(report: BenchmarkReport) -> None:
        lb = report.leaderboard()
        if lb.empty:
            print(f"{Fore.RED}No benchmark results available.{Style.RESET_ALL}")
            return
        print("\nLeaderboard (composite ranking):")
        cols = [
            "Rank",
            "Algorithm",
            report.primary_metric,
            "Time (s)",
            "Scalability Score",
            "Performance Score",
            "Composite Score",
        ]
        view = lb[[c for c in cols if c in lb.columns]].head(15)
        print(view.to_string(index=False))

