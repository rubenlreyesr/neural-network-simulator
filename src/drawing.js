// Crop the ink to a 20-pixel box and center by its intensity-weighted centroid.
export function normalizeDigit(canvas) {
  const { width: w, height: h } = canvas,
    rgba = canvas.getContext("2d").getImageData(0, 0, w, h).data;
  let x0 = w,
    y0 = h,
    x1 = -1,
    y1 = -1;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (rgba[(y * w + x) * 4] > 24) {
        x0 = Math.min(x0, x);
        y0 = Math.min(y0, y);
        x1 = Math.max(x1, x);
        y1 = Math.max(y1, y);
      }
  if (x1 < 0) return null;
  const small = document.createElement("canvas");
  small.width = small.height = 28;
  const ctx = small.getContext("2d");
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, 28, 28);
  const scale = 20 / Math.max(x1 - x0 + 1, y1 - y0 + 1),
    sw = (x1 - x0 + 1) * scale,
    sh = (y1 - y0 + 1) * scale;
  ctx.drawImage(
    canvas,
    x0,
    y0,
    x1 - x0 + 1,
    y1 - y0 + 1,
    (28 - sw) / 2,
    (28 - sh) / 2,
    sw,
    sh,
  );
  const pixels = ctx.getImageData(0, 0, 28, 28).data;
  let mass = 0,
    cx = 0,
    cy = 0;
  for (let i = 0; i < 784; i++) {
    const v = pixels[i * 4];
    mass += v;
    cx += (i % 28) * v;
    cy += Math.floor(i / 28) * v;
  }
  const dx = Math.round(13.5 - cx / mass),
    dy = Math.round(13.5 - cy / mass),
    result = new Float32Array(784);
  for (let y = 0; y < 28; y++)
    for (let x = 0; x < 28; x++)
      if (x + dx >= 0 && x + dx < 28 && y + dy >= 0 && y + dy < 28)
        result[(y + dy) * 28 + x + dx] = pixels[(y * 28 + x) * 4] / 255;
  return result;
}
