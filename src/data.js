import * as tf from "@tensorflow/tfjs";
const ROOT = new URL(
  "mnist/",
  typeof document !== "undefined" ? document.baseURI : "http://localhost/",
).href;
export function parseIDX(buffer, images) {
  const v = new DataView(buffer),
    count = v.getUint32(4),
    offset = images ? 16 : 8;
  if (
    v.getUint32(0) !== (images ? 2051 : 2049) ||
    (images && (v.getUint32(8) !== 28 || v.getUint32(12) !== 28)) ||
    buffer.byteLength !== offset + count * (images ? 784 : 1)
  )
    throw Error("Invalid MNIST data.");
  return new Uint8Array(buffer, offset);
}
async function download(name, images) {
  let response;
  try {
    response = await fetch(ROOT + name, {
      signal: AbortSignal.timeout(120000),
    });
  } catch {
    throw Error(
      "Unable to load MNIST. Check your connection and that public/mnist files were included in the deployment, then retry.",
    );
  }
  if (!response.ok)
    throw Error(
      `MNIST download failed (${response.status}). Check your connection and retry.`,
    );
  // Some static hosts label .gz files Content-Encoding:gzip, so fetch already
  // decompresses them. Detect the bytes instead of decompressing twice.
  let buffer = await response.arrayBuffer();
  const signature = new Uint8Array(buffer, 0, Math.min(2, buffer.byteLength));
  if (signature[0] === 31 && signature[1] === 139)
    buffer = await new Response(
      new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip")),
    ).arrayBuffer();
  return parseIDX(buffer, images);
}
let pending;
export function loadData() {
  if (!pending)
    pending = Promise.all([
      download("train-images-idx3-ubyte.gz", true),
      download("train-labels-idx1-ubyte.gz", false),
      download("t10k-images-idx3-ubyte.gz", true),
      download("t10k-labels-idx1-ubyte.gz", false),
    ])
      .then(([trainImages, trainLabels, testImages, testLabels]) => ({
        trainImages,
        trainLabels,
        testImages,
        testLabels,
      }))
      .catch((e) => {
        pending = null;
        throw e;
      });
  return pending;
}
// Fixed shuffled indices: training and validation never overlap; test is separate.
export function indices(count, seed = 42) {
  const a = Array.from({ length: count }, (_, i) => i);
  let s = seed;
  for (let i = count - 1; i > 0; i--) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const j = Math.floor((s / 4294967296) * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
export function tensors(images, labels, ids) {
  const x = new Float32Array(ids.length * 784),
    y = new Float32Array(ids.length * 10);
  ids.forEach((id, n) => {
    for (let p = 0; p < 784; p++) x[n * 784 + p] = images[id * 784 + p] / 255;
    y[n * 10 + labels[id]] = 1;
  });
  return [
    tf.tensor4d(x, [ids.length, 28, 28, 1]),
    tf.tensor2d(y, [ids.length, 10]),
  ];
}
