const assert = require("node:assert/strict");
const logic = require("../ui/recommendation_ui_logic.js");

function testSingleModelPayload() {
  const out = logic.buildRecommendationPayload({
    userId: "user-1",
    topN: 10,
    strategy: "single_model",
    singleModel: "SVD",
  });
  assert.ok(!out.error);
  assert.equal(out.payload.strategy, "single_model");
  assert.equal(out.payload.algorithm, "SVD");
  assert.equal(out.payload.top_n, 10);
}

function testEnsemblePayloadAndAutoNormalize() {
  const out = logic.buildRecommendationPayload({
    userId: "user-2",
    topN: 5,
    strategy: "ensemble_weighted",
    autoNormalizeWeights: true,
    ensembleModels: [
      { algorithm: "Temporal-SVD", model_id: "m1", selected: true, weight: 35 },
      { algorithm: "BPR", model_id: "m2", selected: true, weight: 25 },
      { algorithm: "ALS", model_id: "m3", selected: true, weight: 20 },
      { algorithm: "SVD", model_id: "m4", selected: true, weight: 10 },
      { algorithm: "Ecommerce-Popularity", model_id: "m5", selected: true, weight: 10 },
    ],
  });
  assert.ok(!out.error);
  assert.equal(out.payload.strategy, "ensemble_weighted");
  assert.equal(out.payload.models.length, 5);
  assert.equal(out.payload.models[0].model_id, "m1");
}

function testDuplicateModelValidation() {
  const valid = logic.validateEnsembleSelection(
    [
      { algorithm: "SVD", selected: true, weight: 50 },
      { algorithm: "svd", selected: true, weight: 50 },
    ],
    true,
  );
  assert.equal(valid.valid, false);
  assert.match(valid.reason, /Duplicate model/i);
}

function testInvalidTotalWithoutAutoNormalize() {
  const out = logic.buildRecommendationPayload({
    userId: "user-3",
    topN: 10,
    strategy: "ensemble_weighted",
    autoNormalizeWeights: false,
    ensembleModels: [
      { algorithm: "SVD", selected: true, weight: 10 },
      { algorithm: "ALS", selected: true, weight: 10 },
    ],
  });
  assert.ok(out.error);
  assert.match(out.error, /Weights must sum to 1.0 or 100.0/i);
}

function testContributionRender() {
  const text = logic.renderContributionText([
    { algorithm: "Temporal-SVD", share_pct: 49.8 },
    { algorithm: "BPR", share_pct: 30.4 },
  ]);
  assert.equal(text, "Temporal-SVD: 49.80% | BPR: 30.40%");
}

function run() {
  testSingleModelPayload();
  testEnsemblePayloadAndAutoNormalize();
  testDuplicateModelValidation();
  testInvalidTotalWithoutAutoNormalize();
  testContributionRender();
  console.log("ui_recommendation_ui_logic tests passed");
}

run();
