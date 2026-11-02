# Neural Network Simulator

A modest, working MNIST playground built with JavaScript, TensorFlow.js 4.22, and Vite. Design a classifier, train it in your browser, compare experiments, and recognize a digit you draw. No server, account, API key, or Python environment is required.

## Review status and working directory

This is an initial implementation for review, not an approved specification. The interface currently uses the working name **Neural Lab**. The implementation choices below were made before a specification review and should be confirmed with the project owner before further feature development.

- Repository: https://github.com/rubenlreyesr/neural-network-simulator
- Authoritative working directory: `C:\Users\Ruben\Documents\Personal\Neural Network Simulator\repo`
- All ongoing development belongs in this checkout.
- Confirm implementation changes with the project owner before further feature development.

## What the system currently does

The application has three areas on one page:

1. **Design:** choose a dense or convolutional preset, edit ordered layer cards, and inspect the resulting tensor shapes and total parameter count.
2. **Train & compare:** choose the sample count, epochs, batch size, and learning rate; train a fresh model on MNIST; compare accuracy and loss with recent runs; optionally evaluate a held-out test sample.
3. **Try a digit:** draw a digit or select a random MNIST test image and inspect the predicted digit and all ten class scores.

A collapsible storage area saves and loads named configurations and trained weights in the current browser. It also imports and exports configuration JSON.

## Implementation choices to review

These are descriptions of the current implementation, not agreed requirements or commitments for future work.

| Area | Current behavior | Review consideration |
| --- | --- | --- |
| Visual editor | Ordered cards with add/remove/up/down buttons | There is no node-and-connection diagram or drag-and-drop canvas. |
| Network structure | Sequential dense and convolutional models, pooling, dropout | No branching, recurrent networks, or custom connections. |
| Architecture inspection | Layer output dimensions and total parameters | No individual neuron, weight, activation, or gradient visualization. |
| Training | MNIST only, Adam optimizer, cross-entropy, fresh weights each run | Decide whether resume, additional optimizers, or other datasets are needed. |
| Experiment comparison | Last five runs, charts and summary table, current session only | Saved weights do not restore experiment history. |
| Data size | 1,000 / 5,000 / 10,000 training images; 1,000 validation and 1,000 test images | Full MNIST training and full test evaluation are not implemented. |
| Persistence | Browser-local configuration and weight saves | No cloud sync or downloadable trained-weight backup. |
| Learning experience | Architecture and learning curves | This is not yet an interactive explanation of backpropagation. |
| Appearance | Responsive green/cream interface, working name Neural Lab | Branding and layout have not been approved. |

## Run locally

Install Node.js 22 or later, then open a terminal in this directory:

```sh
npm ci
npm run dev
```

Open the localhost URL printed by Vite. On Windows PowerShell use `npm.cmd` in place of `npm` if script execution is restricted. If your npm configuration selects a shell that is unavailable, append `--script-shell=cmd.exe` to npm commands on Windows.

```sh
npm test
npm run build
npm run preview
```

The production site is in `dist/`. Serve it over HTTP or HTTPS; do not open `index.html` through `file://`.

## First experiment

1. Keep the dense preset: 784 flattened pixels → 64 ReLU neurons → 10 softmax classes (50,890 parameters).
2. Choose 1,000 images for a quick check, or use the default 5,000. Train for 3 epochs.
3. Watch training and validation accuracy/loss. Solid lines represent validation; dashed lines represent training.
4. Draw one large digit and select **Recognize digit**. Alternatively, select **Use MNIST sample** to inspect an actual test image.
5. Save the weights and configuration before changing the architecture if you want to keep the model.
6. Try the convolution preset with the same training settings. Compare both runs in the charts and table.

Training time depends heavily on the browser, GPU, network architecture, and dataset size. Try the quick setting first. CNN training can take minutes on a CPU. Keep the tab visible; background tabs may be throttled. Stop is checked after a batch; it does not abort an in-progress dataset download or kernel operation.

## Implemented in this MVP

- Information buttons beside every editable setting and the configuration save/load controls, with plain-language examples, ranges, and tradeoffs. Help opens on hover, keyboard focus, or click/tap; Escape dismisses it. Pooling, automatic flattening, and the fixed output layer also have explanations.
- Immediate numeric validation for epochs, neurons, filters, and dropout. Allowed ranges appear beside the fields; blank, fractional integer-only, and out-of-range entries show inline errors and disable training until corrected. Values are not silently rounded or clamped, and typing retains focus.

- Editable, ordered layer cards: dense (1–256 neurons), convolution (1–64 filters, 3/5 kernels), 2×2 max pooling, and dropout (0–0.8).
- ReLU, tanh, sigmoid, and linear activations for hidden dense/convolution layers.
- Add, remove, and reorder layers using buttons. Flatten and the 10-class softmax output are automatic.
- Actual TensorFlow.js model construction, output shapes per layer, and parameter counts. Invalid layer order is explained; networks are capped at 10 editable layers and 2 million parameters.
- Browser training with Adam, categorical cross-entropy, configurable sample count, batch size, learning rate, and epochs.
- Live epoch-level training/validation charts; comparison of the last five runs with architecture, settings, elapsed time, and optional test accuracy.
- Stop after the current batch. Every new training run initializes fresh weights; it does not resume a saved model.
- Pointer/touch digit drawing with cropping, scale normalization, centroid centering, all ten prediction scores, and a test-image alternative.
- Named configuration saves in localStorage; trained topology and weights in IndexedDB, with configuration metadata. Save/load/delete controls.
- Configuration JSON import/export with schema and resource validation. JSON exports do **not** include weights.
- Responsive layout, labeled controls, keyboard-operable layer editor, live status announcements, and an MNIST sample alternative to pointer drawing.
- Automated architecture, dataset parser/split, training, serialization, and tensor-cleanup tests.

## Data and interpretation

The app loads four bundled original compressed MNIST IDX files (~12 MB total) from its own static host. The unchanged files were obtained from the [Google-hosted MNIST mirror](https://storage.googleapis.com/cvdf-datasets/mnist/); provenance and MD5 hashes are in `public/mnist/SOURCE.md`. Include `public/mnist/` in your repository. The standard dataset contains 60,000 training images and 10,000 test images. Files are decoded using the browser's `DecompressionStream`; pixels are converted to floats in [0,1] and labels to ten-class one-hot vectors.

The training IDs are deterministically shuffled with seed 42. Runs use the first 1,000, 5,000, or 10,000 IDs. Validation always uses shuffled positions 50,000–50,999, so it does not overlap any supported training subset. Evaluation uses a fixed shuffled sample of 1,000 images from the separate official test set. This is deliberately a **subset experiment**, not a full-MNIST benchmark. Smaller training sets are nested within larger ones. Weight initialization and training batch order are random, so experiments are not bit-for-bit reproducible. Keep settings equal when comparing architectures, repeat runs, and do not infer an architecture's superiority from a single run.

Use validation results for tuning. Repeatedly consulting the test score can leak information into your choices. The random MNIST sample button is an exploratory tool; some shown examples can overlap the evaluation sample. Your handwriting can differ substantially from MNIST even after normalization. Softmax scores are not calibrated reliability estimates.

Data downloads stay in page memory (and may benefit from normal HTTP caching). Reloading may download them again. No images or weights are sent to a training server. Your static host serves dataset files; Google serves optional web fonts. Ordinary network requests disclose standard connection metadata to those hosts. TensorFlow.js is bundled in the built app.

## Save behavior and browser support

Use a current Chrome, Edge, Firefox, or Safari with WebGL, IndexedDB, `DecompressionStream`, and `crypto.randomUUID`. TensorFlow.js chooses an available backend; the header reports it. CPU fallback can be slow. Storage can fail in private browsing or when quota is exhausted; errors appear in the status bar.

Saves belong to a browser profile and origin (scheme + hostname + port). Moving from localhost to GitHub Pages does not move saves. Repositories under the same `username.github.io` origin share browser storage; save names are application-wide. Clearing site data removes weights and configurations. There is no cross-device sync or downloadable weights backup in this MVP. Comparison history is page-session-only and is not restored with saved weights. Loading a saved model restores inference; selecting Train creates a new model. Architecture edits discard active weights, while explicitly saved models remain available.

## Deploy to GitHub Pages

1. The checkout is connected to `https://github.com/rubenlreyesr/neural-network-simulator.git`. The source, `package-lock.json`, and `.github/workflows/deploy.yml` are already at its root. Review and approve the implementation before committing or pushing it. Do not include `node_modules/` or `dist/` in the source repository.
2. Use `main` as the default branch, or update the workflow's branch filter to your branch name.
3. In the repository, open **Settings → Pages → Build and deployment → Source → GitHub Actions**.
4. Push your source to `main` (or manually run **Deploy to GitHub Pages** from the Actions tab).
5. The workflow installs locked dependencies, runs tests, builds `dist/`, uploads the Pages artifact, and deploys it. Open the URL shown by the deployment.

`vite.config.js` sets `base: './'`, so assets resolve on both `https://USERNAME.github.io/REPOSITORY/` and custom domains. There are no client-side path routes to configure. The app has no production server, secrets, or environment variables. Pushing to `main` triggers the included deployment workflow. GitHub Pages must be configured to use GitHub Actions before deployment can succeed.

The workflow follows [GitHub's custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Source map

| File                           | Purpose                                                          |
| ------------------------------ | ---------------------------------------------------------------- |
| `src/main.js`                  | Editor, training lifecycle, charts, prediction, and save/load UI |
| `src/model.js`                 | Model validation, presets, construction, and disposal            |
| `src/data.js`                  | IDX download/parse, deterministic splits, tensor conversion      |
| `src/drawing.js`               | Handwritten digit normalization                                  |
| `src/style.css`                | Responsive interface                                             |
| `tests/model.test.js`          | CPU-based automated functional tests                             |
| `.github/workflows/deploy.yml` | GitHub Pages build and deployment                                |

See the [TensorFlow.js API](https://js.tensorflow.org/api/latest/) and [save/load guide](https://www.tensorflow.org/js/guide/save_load) for the model APIs.

## Future enhancements — not implemented

Drag-and-drop graph editing and branching networks; arbitrary datasets or image uploads; full 60,000-image training; configurable optimizers and seeds; learning-rate schedules; early stopping; training resume; confusion matrices; per-class metrics; downloadable trained-model bundles; import of arbitrary TensorFlow.js models; persistent run histories; weight/activation visualizations; step-by-step backpropagation; Web Workers; offline dataset caching; mobile performance tuning.

The app visualizes architecture and learning curves. It does not visualize individual gradients or explain every mathematical step of backpropagation.
