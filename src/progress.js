// Counts completed batches, including a final batch smaller than batchSize.
export function batchProgress(batchIndex, batchSize, samples) {
  return {
    batches: Math.min(batchIndex + 1, Math.ceil(samples / batchSize)),
    totalBatches: Math.ceil(samples / batchSize),
    images: Math.min((batchIndex + 1) * batchSize, samples),
  };
}

// Explicit validation lets the UI paint its phase before evaluation begins.
export async function validationMetrics(model, data, batchSize) {
  const results = model.evaluate(...data, { batchSize });
  try {
    const [loss, accuracy] = await Promise.all(results.map((t) => t.data()));
    return { val_loss: loss[0], val_acc: accuracy[0] };
  } finally {
    results.forEach((t) => t.dispose());
  }
}
