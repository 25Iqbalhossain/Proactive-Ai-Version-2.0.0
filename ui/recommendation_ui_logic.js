/* Frontend recommendation payload/validation helpers. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.RecUiLogic = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  function _toNumber(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : NaN;
  }

  function validateEnsembleSelection(models, autoNormalizeWeights, minModels = 2, maxModels = 5) {
    const selected = (models || []).filter((m) => m && m.selected);
    if (selected.length < minModels) {
      return { valid: false, total: 0, reason: `Select at least ${minModels} models.`, selectedCount: selected.length };
    }
    if (selected.length > maxModels) {
      return { valid: false, total: 0, reason: `Select at most ${maxModels} models.`, selectedCount: selected.length };
    }

    const names = new Set();
    let total = 0;
    for (const model of selected) {
      const name = String(model.algorithm || "").trim();
      const weight = _toNumber(model.weight);
      if (!name) {
        return { valid: false, total, reason: "Model name is required.", selectedCount: selected.length };
      }
      if (names.has(name.toLowerCase())) {
        return { valid: false, total, reason: `Duplicate model '${name}' is not allowed.`, selectedCount: selected.length };
      }
      names.add(name.toLowerCase());
      if (!(weight > 0)) {
        return { valid: false, total, reason: `Weight for '${name}' must be positive.`, selectedCount: selected.length };
      }
      total += weight;
    }

    const nearOne = Math.abs(total - 1.0) < 1e-8;
    const nearHundred = Math.abs(total - 100.0) < 1e-8;
    if (!(nearOne || nearHundred) && !autoNormalizeWeights) {
      return {
        valid: false,
        total,
        reason: `Weights must sum to 1.0 or 100.0. Current total: ${total.toFixed(4)}.`,
        selectedCount: selected.length,
      };
    }

    return {
      valid: true,
      total,
      reason: (nearOne || nearHundred) ? "Weights are valid." : "Weights will be auto-normalized.",
      selectedCount: selected.length,
      normalizationNeeded: !(nearOne || nearHundred),
    };
  }

  function buildRecommendationPayload(input) {
    const userId = String(input.userId || "").trim();
    const topN = _toNumber(input.topN);
    const strategy = String(input.strategy || "best_promoted_model").trim();
    if (!userId) return { error: "User ID is required." };
    if (!(topN === 5 || topN === 10)) return { error: "Top-N must be 5 or 10." };

    const payload = {
      user_id: userId,
      top_n: topN,
      strategy,
    };

    if (strategy === "single_model") {
      const algorithm = String(input.singleModel || "").trim();
      if (!algorithm) return { error: "A model must be selected for single_model strategy." };
      payload.algorithm = algorithm;
      return { payload };
    }

    if (strategy === "ensemble_weighted") {
      const autoNormalizeWeights = !!input.autoNormalizeWeights;
      const validation = validateEnsembleSelection(input.ensembleModels || [], autoNormalizeWeights);
      if (!validation.valid) return { error: validation.reason };
      payload.models = (input.ensembleModels || [])
        .filter((m) => m.selected)
        .map((m) => ({
          algorithm: String(m.algorithm).trim(),
          weight: _toNumber(m.weight),
          model_id: m.model_id ? String(m.model_id).trim() : undefined,
        }));
      payload.auto_normalize_weights = autoNormalizeWeights;
      return { payload, validation };
    }

    // best_promoted_model by default
    return { payload };
  }

  function renderContributionText(contributions) {
    return (contributions || [])
      .map((c) => `${c.algorithm}: ${Number(c.share_pct || 0).toFixed(2)}%`)
      .join(" | ");
  }

  return {
    validateEnsembleSelection,
    buildRecommendationPayload,
    renderContributionText,
  };
});
