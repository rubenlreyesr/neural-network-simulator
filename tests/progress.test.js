import test from "node:test";
import assert from "node:assert/strict";
import * as tf from "@tensorflow/tfjs";
import { batchProgress, validationMetrics } from "../src/progress.js";
import { buildModel, disposeModel } from "../src/model.js";

test("batch progress counts a partial final batch accurately", () => {
  assert.deepEqual(batchProgress(-1, 64, 1000), { batches: 0, totalBatches: 16, images: 0 });
  assert.deepEqual(batchProgress(14, 64, 1000), { batches: 15, totalBatches: 16, images: 960 });
  assert.deepEqual(batchProgress(15, 64, 1000), { batches: 16, totalBatches: 16, images: 1000 });
});

test("explicit validation matches fit validation and supports stopped runs", async () => {
  await tf.setBackend("cpu");
  const model = buildModel({ version: 1, layers: [] });
  model.compile({ optimizer: tf.train.sgd(0.01), loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  const x = tf.zeros([10, 28, 28, 1]);
  const y = tf.tidy(() => tf.oneHot(tf.zeros([10], "int32"), 10));
  try {
    let original;
    await model.fit(x, y, { epochs: 1, batchSize: 4, validationData: [x, y], callbacks: {
      onEpochEnd: async (_, logs) => { original = logs; },
    } });
    const before = tf.memory().numTensors;
    const actual = await validationMetrics(model, [x, y], 4);
    assert.ok(Math.abs(original.val_loss - actual.val_loss) < 1e-6);
    assert.equal(actual.val_acc, original.val_acc);
    assert.equal(tf.memory().numTensors, before);
    const batches = [];
    await model.fit(x, y, { epochs: 2, batchSize: 4, callbacks: {
      onBatchEnd: async (i) => { batches.push(batchProgress(i, 4, 10)); model.stopTraining = true; },
      onEpochEnd: async (_, logs) => { Object.assign(logs, await validationMetrics(model, [x, y], 4)); assert.ok(Number.isFinite(logs.val_loss)); },
    } });
    assert.deepEqual(batches, [{ batches: 1, totalBatches: 3, images: 4 }]);
  } finally {
    x.dispose(); y.dispose(); disposeModel(model);
  }
});
