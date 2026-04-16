"""
insights/explainability.py – Recommendation explainability engine

Explains WHY an item was recommended to a user.
Techniques:
  - Item similarity analysis (items the user liked → similar items)
  - User similarity analysis (similar users liked this item)
  - Feature importance (which latent dimensions drove the score)
  - Popularity-based explanations
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from data_processing.interaction_matrix import InteractionMatrix
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Explanation:
    """Structured explanation for a single recommendation."""
    user_id:       str
    item_id:       str
    score:         float
    primary_reason: str
    supporting_items: list[dict]   # items user liked that led to this rec
    similar_users:    list[dict]   # similar users who interacted with this item
    feature_importance: list[dict] # top latent dimensions
    confidence:    str             # HIGH | MEDIUM | LOW


class ExplainabilityEngine:
    """
    Generates human-readable explanations for recommendations.

    Usage
    -----
    engine = ExplainabilityEngine(interaction_matrix, item_factors, user_factors)
    exp    = engine.explain("user_123", "item_456", score=0.87)
    print(exp.primary_reason)
    """

    def __init__(
        self,
        im:           InteractionMatrix,
        item_factors: Optional[np.ndarray] = None,   # (n_items, n_factors)
        user_factors: Optional[np.ndarray] = None,   # (n_users, n_factors)
        item_metadata: Optional[pd.DataFrame] = None, # optional item titles/categories
    ):
        self.im           = im
        self.item_factors = item_factors
        self.user_factors = user_factors
        self.item_metadata = item_metadata

    def explain(
        self,
        user_id: str,
        item_id: str,
        score: float = 0.0,
        n_similar_items: int = 3,
        n_similar_users: int = 3,
    ) -> Explanation:
        """
        Generate a full explanation for why item_id was recommended to user_id.
        """
        u_idx = self.im.user_index.get(str(user_id))
        i_idx = self.im.item_index.get(str(item_id))

        supporting_items   = []
        similar_users      = []
        feature_importance = []

        if u_idx is not None and i_idx is not None:
            supporting_items = self._similar_items_in_history(
                u_idx, i_idx, n_similar_items
            )
            similar_users = self._similar_users_for_item(
                u_idx, i_idx, n_similar_users
            )
            if self.item_factors is not None and self.user_factors is not None:
                feature_importance = self._feature_importance(u_idx, i_idx)

        primary_reason = self._build_primary_reason(
            user_id, item_id, supporting_items, similar_users, score
        )
        confidence = self._confidence(score, len(supporting_items), len(similar_users))

        log.debug("Explained | user=%s item=%s reason=%s", user_id, item_id, primary_reason)
        return Explanation(
            user_id           = user_id,
            item_id           = item_id,
            score             = score,
            primary_reason    = primary_reason,
            supporting_items  = supporting_items,
            similar_users     = similar_users,
            feature_importance = feature_importance,
            confidence        = confidence,
        )

    def explain_batch(
        self,
        user_id: str,
        recommendations: list[dict],
    ) -> list[Explanation]:
        """Explain a list of recommendations for one user."""
        return [
            self.explain(user_id, rec["item_id"], score=rec.get("score", 0.0))
            for rec in recommendations
        ]

    # ── Item similarity analysis ───────────────────────────────────────────────

    def _similar_items_in_history(
        self, u_idx: int, i_idx: int, n: int
    ) -> list[dict]:
        """
        Find items in the user's history that are most similar to item_id.
        Uses cosine similarity of item latent factors if available,
        otherwise falls back to co-occurrence.
        """
        if self.item_factors is None:
            return self._cooccurrence_items(u_idx, i_idx, n)

        target_vec   = self.item_factors[i_idx]
        target_norm  = np.linalg.norm(target_vec) + 1e-9

        # Items the user has interacted with
        seen_indices = self.im.matrix[u_idx].nonzero()[1]
        if len(seen_indices) == 0:
            return []

        sims = []
        for j in seen_indices:
            if j == i_idx: continue
            v    = self.item_factors[j]
            sim  = float(np.dot(target_vec, v) / (np.linalg.norm(v) * target_norm + 1e-9))
            sims.append((j, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "item_id":   self.im.item_ids[j],
                "similarity": round(sim, 4),
                "title":     self._item_title(self.im.item_ids[j]),
            }
            for j, sim in sims[:n]
        ]

    def _cooccurrence_items(self, u_idx, i_idx, n) -> list[dict]:
        """Fallback: items seen by users who also interacted with item_id."""
        item_col   = self.im.matrix[:, i_idx].nonzero()[0]
        user_row   = self.im.matrix[u_idx].nonzero()[1]
        overlap    = set(item_col) & {u_idx}
        seen_items = list(user_row)[:n]
        return [
            {"item_id": self.im.item_ids[j], "similarity": None, "title": self._item_title(self.im.item_ids[j])}
            for j in seen_items
        ]

    # ── User similarity analysis ───────────────────────────────────────────────

    def _similar_users_for_item(
        self, u_idx: int, i_idx: int, n: int
    ) -> list[dict]:
        """
        Find users similar to the target user who also interacted with item_id.
        """
        if self.user_factors is None:
            # Fallback: just return users who interacted with item_id
            users_who_interacted = self.im.matrix[:, i_idx].nonzero()[0]
            return [
                {"user_id": self.im.user_ids[u], "similarity": None}
                for u in users_who_interacted[:n]
                if u != u_idx
            ]

        target_user  = self.user_factors[u_idx]
        target_norm  = np.linalg.norm(target_user) + 1e-9

        # Only compute over users who interacted with item_id
        users_with_item = self.im.matrix[:, i_idx].nonzero()[0]
        sims = []
        for u in users_with_item:
            if u == u_idx: continue
            v   = self.user_factors[u]
            sim = float(np.dot(target_user, v) / (np.linalg.norm(v) * target_norm + 1e-9))
            sims.append((u, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        return [
            {"user_id": self.im.user_ids[u], "similarity": round(sim, 4)}
            for u, sim in sims[:n]
        ]

    # ── Feature importance ─────────────────────────────────────────────────────

    def _feature_importance(self, u_idx: int, i_idx: int, top_k: int = 5) -> list[dict]:
        """
        Compute element-wise product of user and item latent vectors.
        The top dimensions indicate which latent features drove the score.
        """
        u_vec   = self.user_factors[u_idx]
        i_vec   = self.item_factors[i_idx]
        contrib = u_vec * i_vec
        top_idx = np.argsort(np.abs(contrib))[::-1][:top_k]
        return [
            {"dimension": int(d), "contribution": round(float(contrib[d]), 4)}
            for d in top_idx
        ]

    # ── Primary reason builder ─────────────────────────────────────────────────

    @staticmethod
    def _build_primary_reason(user_id, item_id, supporting_items, similar_users, score) -> str:
        if supporting_items:
            titles = [s.get("title") or s.get("item_id") for s in supporting_items[:2]]
            return f"Recommended because you interacted with similar items: {', '.join(titles)}"
        if similar_users:
            return (
                f"Users with similar taste to you also interacted with this item "
                f"({len(similar_users)} similar users found)"
            )
        if score > 0.7:
            return "Highly rated item that matches your interaction patterns"
        return "Recommended based on your overall interaction history"

    @staticmethod
    def _confidence(score: float, n_items: int, n_users: int) -> str:
        if n_items >= 2 or n_users >= 2:  return "HIGH"
        if n_items >= 1 or n_users >= 1:  return "MEDIUM"
        return "LOW"

    def _item_title(self, item_id: str) -> Optional[str]:
        if self.item_metadata is None: return None
        row = self.item_metadata[self.item_metadata["itemID"].astype(str) == item_id]
        if row.empty: return None
        for col in ("title", "name", "description"):
            if col in row.columns:
                return str(row.iloc[0][col])
        return None
