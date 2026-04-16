"""
General-purpose neighborhood and linear recommendation models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import TOP_K
from data_processing.interaction_matrix import InteractionMatrix, InteractionMatrixBuilder
from utils.logger import get_logger

log = get_logger(__name__)

_builder = InteractionMatrixBuilder()


def _rating_preds(score_matrix: np.ndarray, im: InteractionMatrix, test: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in test.iterrows():
        uid = str(row["userID"])
        iid = str(row["itemID"])
        u_idx = im.user_index.get(uid)
        i_idx = im.item_index.get(iid)
        if u_idx is not None and i_idx is not None:
            records.append({"userID": uid, "itemID": iid, "prediction": float(score_matrix[u_idx, i_idx])})
    return pd.DataFrame(records)


def _ranking_preds(
    score_matrix: np.ndarray,
    im: InteractionMatrix,
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    target_users = test["userID"].astype(str).unique().tolist()
    return _builder.score_matrix_to_ranking_df(
        im,
        score_matrix,
        train,
        top_k,
        target_user_ids=target_users,
    )


def _prune_top_k(similarity: np.ndarray, k_neighbors: int) -> np.ndarray:
    if similarity.shape[0] <= 1:
        return similarity

    pruned = similarity.copy()
    np.fill_diagonal(pruned, 0.0)
    k_neighbors = max(1, min(int(k_neighbors), pruned.shape[0] - 1))
    if k_neighbors >= pruned.shape[0] - 1:
        return pruned

    thresholds = np.partition(pruned, -k_neighbors, axis=1)[:, -k_neighbors]
    return np.where(pruned >= thresholds[:, None], pruned, 0.0)


def run_user_knn(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = TOP_K,
    k_neighbors: int = 40,
    shrinkage: float = 10.0,
    return_model: bool = False,
) -> tuple:
    """
    User-based k-nearest neighbors collaborative filtering.
    """
    im = _builder.build(train)
    dense = im.matrix.toarray().astype(np.float32)

    user_sim = cosine_similarity(dense)
    if shrinkage > 0:
        counts = (dense > 0).astype(np.float32) @ (dense > 0).astype(np.float32).T
        user_sim *= counts / (counts + float(shrinkage))
    user_sim = _prune_top_k(user_sim, k_neighbors)

    denom = np.abs(user_sim).sum(axis=1, keepdims=True) + 1e-9
    score_matrix = (user_sim @ dense) / denom

    log.info("User-KNN trained | neighbors=%d shrinkage=%.2f", k_neighbors, shrinkage)

    rating_preds = _rating_preds(score_matrix, im, test)
    ranking_preds = _ranking_preds(score_matrix, im, train, test, top_k)

    if return_model:
        return rating_preds, ranking_preds, {
            "type": "user_knn",
            "score_matrix": score_matrix.astype(np.float32),
            "interaction_matrix": im,
        }
    return rating_preds, ranking_preds


def run_item_knn(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = TOP_K,
    k_neighbors: int = 40,
    shrinkage: float = 10.0,
    return_model: bool = False,
) -> tuple:
    """
    Item-based k-nearest neighbors collaborative filtering.
    """
    im = _builder.build(train)
    dense = im.matrix.toarray().astype(np.float32)

    item_sim = cosine_similarity(dense.T)
    if shrinkage > 0:
        counts = (dense > 0).astype(np.float32).T @ (dense > 0).astype(np.float32)
        item_sim *= counts / (counts + float(shrinkage))
    item_sim = _prune_top_k(item_sim, k_neighbors)

    denom = np.abs(item_sim).sum(axis=1, keepdims=True).T + 1e-9
    score_matrix = (dense @ item_sim) / denom

    log.info("Item-KNN trained | neighbors=%d shrinkage=%.2f", k_neighbors, shrinkage)

    rating_preds = _rating_preds(score_matrix, im, test)
    ranking_preds = _ranking_preds(score_matrix, im, train, test, top_k)

    if return_model:
        return rating_preds, ranking_preds, {
            "type": "item_knn",
            "score_matrix": score_matrix.astype(np.float32),
            "item_sim": item_sim.astype(np.float32),
            "interaction_matrix": im,
        }
    return rating_preds, ranking_preds


def run_ease(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = TOP_K,
    l2_penalty: float = 300.0,
    return_model: bool = False,
) -> tuple:
    """
    EASE: a linear recommender solved in closed form.
    """
    im = _builder.build(train)
    x = im.matrix.toarray().astype(np.float64)

    gram = x.T @ x
    diag = np.arange(gram.shape[0])
    gram[diag, diag] += float(l2_penalty)

    precision = np.linalg.inv(gram)
    coefficients = precision / (-np.diag(precision)[None, :])
    np.fill_diagonal(coefficients, 0.0)

    score_matrix = x @ coefficients

    log.info("EASE trained | items=%d lambda=%.2f", im.n_items, l2_penalty)

    rating_preds = _rating_preds(score_matrix, im, test)
    ranking_preds = _ranking_preds(score_matrix, im, train, test, top_k)

    if return_model:
        return rating_preds, ranking_preds, {
            "type": "ease",
            "coefficients": coefficients.astype(np.float32),
            "interaction_matrix": im,
        }
    return rating_preds, ranking_preds
