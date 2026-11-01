# Verification — November 1, 2026

## Passed

- Production Vite build; dependency audit reported zero vulnerabilities at installation.
- Six automated test groups: model counts/probabilities, invalid architecture rejection, deterministic non-overlapping data splits, IDX validation, all four bundled MNIST file hashes/shapes, and actual dense/CNN training plus serialization round trips and tensor cleanup.
- Browser WebGL training on real MNIST for both presets (1,000 training images, 1,000 validation images, three epochs).
- Dense smoke run: 78.8% validation accuracy and 80.3% accuracy on the fixed 1,000-image test subset. CNN smoke run: 55.9% validation accuracy. These are individual random runs, not benchmarks or expected outcomes.
- Sample inference; browser IndexedDB save and reload restored the same sample prediction (digit 0, 97.8% score).
- Pointer drawing and normalization: a hand-drawn vertical stroke was classified as digit 1.
- Production app and bundled dataset loaded under `/neural-lab/`, a repository-style subpath.

## Limits

- Not deployed to GitHub Pages; the included workflow and instructions are ready for your repository.
- Browser smoke checks used the desktop app's Chromium-based browser. Other browsers and physical touch devices were not tested.
- Vite warns that the JavaScript bundle is large because it includes TensorFlow.js (~1.62 MB uncompressed, ~258 KB gzip). This does not prevent building or deployment. MNIST adds ~12 MB of static data, fetched when training/evaluating first requires it.
- No long-running stress test, full accessibility audit, or full-dataset benchmark was performed.
