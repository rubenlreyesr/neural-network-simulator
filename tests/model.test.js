import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { createHash } from "node:crypto";
import * as tf from "@tensorflow/tfjs";
import { presets, validate, buildModel, disposeModel } from "../src/model.js";
import { indices, parseIDX } from "../src/data.js";
test("preset parameter counts and output probability shape", async () => {
  for (const layers of Object.values(presets)) {
    const config = { version: 1, layers },
      m = buildModel(config);
    assert.equal(m.countParams(), validate(config));
    const x = tf.zeros([2, 28, 28, 1]),
      y = m.predict(x);
    assert.deepEqual(y.shape, [2, 10]);
    const p = await y.data();
    assert.ok(Math.abs(p.slice(0, 10).reduce((a, b) => a + b, 0) - 1) < 1e-5);
    x.dispose();
    y.dispose();
    disposeModel(m);
  }
  assert.equal(validate({ version: 1, layers: presets.dense }), 50890);
});
test("rejects invalid architecture and resource-heavy models", () => {
  for (const layers of [
    [
      ...presets.dense,
      { type: "conv", filters: 8, kernel: 3, activation: "relu" },
    ],
    Array(6).fill({ type: "pool" }),
    [{ type: "dense", units: -2, activation: "relu" }],
    [
      { type: "conv", filters: 64, kernel: 3, activation: "relu" },
      { type: "dense", units: 256, activation: "relu" },
    ],
  ])
    assert.throws(() => validate({ version: 1, layers }));
  assert.throws(() => validate({ version: 2, layers: [] }));
});
test("fixed data split is deterministic and non-overlapping", () => {
  const ids = indices(60000);
  assert.deepEqual(ids, indices(60000));
  assert.equal(new Set(ids).size, 60000);
  const training = new Set(ids.slice(0, 10000));
  assert.ok(ids.slice(50000, 51000).every((i) => !training.has(i)));
});
test("IDX parser validates format", () => {
  const b = new ArrayBuffer(9),
    v = new DataView(b);
  v.setUint32(0, 2049);
  v.setUint32(4, 1);
  v.setUint8(8, 7);
  assert.deepEqual(Array.from(parseIDX(b, false)), [7]);
  assert.throws(() => parseIDX(b, true));
});
test("bundled original MNIST files have correct hashes, shapes, and label range", () => {
  for (const [name, hash, count, images] of [
    [
      "train-images-idx3-ubyte.gz",
      "f68b3c2dcbeaaa9fbdd348bbdeb94873",
      60000,
      true,
    ],
    [
      "train-labels-idx1-ubyte.gz",
      "d53e105ee54ea40749a09fcbcd1e9432",
      60000,
      false,
    ],
    [
      "t10k-images-idx3-ubyte.gz",
      "9fb629c4189551a2d022fa330f9573f3",
      10000,
      true,
    ],
    [
      "t10k-labels-idx1-ubyte.gz",
      "ec29112dd5afa0611ce80d1b7f02629c",
      10000,
      false,
    ],
  ]) {
    const compressed = readFileSync(
      new URL("../public/mnist/" + name, import.meta.url),
    );
    assert.equal(createHash("md5").update(compressed).digest("hex"), hash);
    const raw = gunzipSync(compressed),
      parsed = parseIDX(
        raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength),
        images,
      );
    assert.equal(parsed.length, count * (images ? 784 : 1));
    if (!images) assert.ok(parsed.every((x) => x < 10));
  }
});
test("dense and CNN actually train, serialize, reload and release tensors", async () => {
  for (const layers of Object.values(presets)) {
    const baseline = tf.memory().numTensors,
      m = buildModel({ version: 1, layers });
    m.compile({
      optimizer: tf.train.adam(0.01),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });
    const x = tf.ones([4, 28, 28, 1]),
      y = tf.tensor2d(
        Array.from({ length: 4 }, () => [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
      );
    const h = await m.fit(x, y, {
      epochs: 2,
      batchSize: 4,
      validationData: [x, y],
    });
    assert.ok(h.history.loss.every(Number.isFinite));
    assert.ok(h.history.loss[1] < h.history.loss[0]);
    let artifacts;
    await m.save(
      tf.io.withSaveHandler(async (a) => {
        artifacts = a;
        return {
          modelArtifactsInfo: {
            dateSaved: new Date(),
            modelTopologyType: "JSON",
          },
        };
      }),
    );
    const reloaded = await tf.loadLayersModel(tf.io.fromMemory(artifacts));
    const a = m.predict(x),
      b = reloaded.predict(x);
    assert.deepEqual(await a.array(), await b.array());
    tf.dispose([x, y, a, b]);
    disposeModel(m);
    disposeModel(reloaded);
    assert.equal(tf.memory().numTensors, baseline);
  }
});
