"""
utils/metrics.py – Evaluation metrics for explicit and implicit feedback

Explicit:
    RMSE, MAE

Implicit / Ranking:
    Precision@K, Recall@K, NDCG@K, MAP@K
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Explicit feedback metrics
# ══════════════════════════════════════════════════════════════════════════════

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def evaluate_explicit(
    test: pd.DataFrame,
    rating_preds: pd.DataFrame,
) -> dict:
    """
    Compute RMSE and MAE from rating predictions.

    Parameters
    ----------
    test         : ground-truth DataFrame with columns [userID, itemID, rating]
    rating_preds : predicted DataFrame with columns [userID, itemID, prediction]
    """
    if rating_preds is None or rating_preds.empty:
        return {"RMSE": None, "MAE": None}

    merged = test.merge(
        rating_preds[["userID", "itemID", "prediction"]],
        on=["userID", "itemID"],
        how="inner",
    )
    if merged.empty:
        return {"RMSE": None, "MAE": None}

    return {
        "RMSE": round(rmse(merged["rating"].values, merged["prediction"].values), 4),
        "MAE":  round(mae(merged["rating"].values,  merged["prediction"].values), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Implicit / ranking metrics
# ══════════════════════════════════════════════════════════════════════════════

def _dcg(relevances: list) -> float:
    """Discounted Cumulative Gain."""
    return sum(r / np.log2(i + 2) for i, r in enumerate(relevances))


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if not recommended:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not recommended or not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    if not recommended or not relevant:
        return 0.0
    gains  = [1 if item in relevant else 0 for item in recommended[:k]]
    ideal  = [1] * min(len(relevant), k)
    dcg    = _dcg(gains)
    idcg   = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def ap_at_k(recommended: list, relevant: set, k: int) -> float:
    """Average Precision@K."""
    if not recommended or not relevant:
        return 0.0
    hits, score = 0, 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits  += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)


def evaluate_ranking(
    test: pd.DataFrame,
    ranking_preds: pd.DataFrame,
    k: int = 10,
) -> dict:
    """
    Compute Precision@K, Recall@K, NDCG@K, MAP@K from ranking predictions.

    Parameters
    ----------
    test          : ground-truth DataFrame [userID, itemID, rating]
    ranking_preds : ranked recommendations [userID, itemID, score]
    k             : top-K cutoff
    """
    if ranking_preds is None or ranking_preds.empty:
        return {"Precision@K": None, "Recall@K": None, "NDCG@K": None, "MAP@K": None}

    # Build ground-truth sets per user
    relevant_map: dict[str, set] = (
        test.groupby("userID")["itemID"]
            .apply(set)
            .to_dict()
    )

    precisions, recalls, ndcgs, aps = [], [], [], []

    for user_id, group in ranking_preds.groupby("userID"):
        relevant = relevant_map.get(user_id, set())
        if not relevant:
            continue
        recommended = group.sort_values("score", ascending=False)["itemID"].tolist()
        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))
        aps.append(ap_at_k(recommended, relevant, k))

    if not precisions:
        return {"Precision@K": None, "Recall@K": None, "NDCG@K": None, "MAP@K": None}

    return {
        "Precision@K": round(float(np.mean(precisions)), 4),
        "Recall@K":    round(float(np.mean(recalls)),    4),
        "NDCG@K":      round(float(np.mean(ndcgs)),      4),
        "MAP@K":       round(float(np.mean(aps)),         4),
    }


def evaluate_all(
    test: pd.DataFrame,
    rating_preds: Optional[pd.DataFrame],
    ranking_preds: Optional[pd.DataFrame],
    algorithm_name: str,
    k: int = 10,
) -> dict:
    """
    Run both explicit and ranking metrics and merge into one result dict.
    """
    result = {"Algorithm": algorithm_name}
    result.update(evaluate_explicit(test, rating_preds))
    result.update(evaluate_ranking(test, ranking_preds, k))
    return result
