import * as tf from "@tensorflow/tfjs";
export const presets = {
  dense: [{ type: "dense", units: 64, activation: "relu" }],
  cnn: [
    { type: "conv", filters: 8, kernel: 3, activation: "relu" },
    { type: "pool" },
    { type: "conv", filters: 16, kernel: 3, activation: "relu" },
    { type: "pool" },
    { type: "dense", units: 32, activation: "relu" },
  ],
};
const integer = (x, min, max) => Number.isInteger(x) && x >= min && x <= max;
export function validate(config) {
  if (
    config?.version !== 1 ||
    !Array.isArray(config.layers) ||
    config.layers.length > 10
  )
    throw Error("Expected a version 1 configuration with at most 10 layers.");
  let spatial = true,
    size = 28,
    channels = 1,
    width = 784,
    total = 0;
  for (const l of config.layers) {
    if (!["dense", "conv", "pool", "dropout"].includes(l.type))
      throw Error("Unsupported layer type.");
    if (
      ["dense", "conv"].includes(l.type) &&
      !["relu", "tanh", "sigmoid", "linear"].includes(l.activation)
    )
      throw Error("Unsupported activation.");
    if (l.type === "conv") {
      if (!spatial) throw Error("Convolution must come before dense layers.");
      if (!integer(l.filters, 1, 64) || ![3, 5].includes(l.kernel))
        throw Error("Filters must be 1–64; kernels must be 3 or 5.");
      total += (l.kernel * l.kernel * channels + 1) * l.filters;
      channels = l.filters;
      width = size * size * channels;
    }
    if (l.type === "pool") {
      if (!spatial || size < 2)
        throw Error("Pooling needs a spatial input at least 2 × 2.");
      size = Math.floor(size / 2);
      width = size * size * channels;
    }
    if (l.type === "dense") {
      if (!integer(l.units, 1, 256)) throw Error("Dense units must be 1–256.");
      total += (width + 1) * l.units;
      width = l.units;
      spatial = false;
    }
    if (
      l.type === "dropout" &&
      (!Number.isFinite(l.rate) || l.rate < 0 || l.rate > 0.8)
    )
      throw Error("Dropout must be between 0 and 0.8.");
  }
  total += (width + 1) * 10;
  if (total > 2000000)
    throw Error(
      "This MVP limits networks to 2 million parameters. Add pooling before dense layers.",
    );
  return total;
}
export function buildModel(config) {
  validate(config);
  const model = tf.sequential();
  let spatial = true;
  const add = (layer) => model.add(layer);
  const input = () => (model.layers.length ? {} : { inputShape: [28, 28, 1] });
  for (const l of config.layers) {
    if (l.type === "conv")
      add(
        tf.layers.conv2d({
          ...input(),
          filters: l.filters,
          kernelSize: l.kernel,
          padding: "same",
          activation: l.activation,
        }),
      );
    if (l.type === "pool")
      add(tf.layers.maxPooling2d({ ...input(), poolSize: 2, strides: 2 }));
    if (l.type === "dropout")
      add(tf.layers.dropout({ ...input(), rate: l.rate }));
    if (l.type === "dense") {
      if (spatial) {
        add(tf.layers.flatten(input()));
        spatial = false;
      }
      add(tf.layers.dense({ units: l.units, activation: l.activation }));
    }
  }
  if (spatial) add(tf.layers.flatten(input()));
  add(tf.layers.dense({ units: 10, activation: "softmax" }));
  return model;
}
export function disposeModel(model) {
  if (model) {
    model.optimizer?.dispose();
    model.dispose();
  }
}
