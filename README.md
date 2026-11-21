# Neural Network Simulator

**Neural Lab** is a browser-based playground for building and comparing handwritten-digit classifiers. Edit a dense or convolutional neural network, train it on MNIST, inspect its learning curves, and try it on a digit you draw.

The application uses **JavaScript, TensorFlow.js 4.22, and Vite**. Computation runs in your browser. It needs no Python environment, account, API key, or application backend, and its production build can be hosted on GitHub Pages.

- [Repository](https://github.com/rubenlreyesr/neural-network-simulator)
- [GitHub Pages address](https://rubenlreyesr.github.io/neural-network-simulator/)
- [Build and deployment runs](https://github.com/rubenlreyesr/neural-network-simulator/actions)

This README describes the implemented MVP, including its limits. It is not a specification for features that have not been built. The working name and interface remain open to review; confirm further feature changes with the project owner.

## Contents

- [Implemented features](#implemented-features)
- [Run locally](#run-locally)
- [Your first experiment](#your-first-experiment)
- [Designing a network](#designing-a-network)
- [Training settings](#training-settings)
- [Understanding results](#understanding-results)
- [Drawing and prediction](#drawing-and-prediction)
- [Saving and loading](#saving-and-loading)
- [Configuration JSON](#configuration-json)
- [MNIST data and experiment design](#mnist-data-and-experiment-design)
- [GPU, NPU, browser support, and privacy](#gpu-npu-browser-support-and-privacy)
- [GitHub Pages deployment](#github-pages-deployment)
- [Repository guide](#repository-guide)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Future enhancements — not implemented](#future-enhancements--not-implemented)

## Implemented features

The page has three main areas: **Design**, **Train & compare**, and **Try a digit**, plus configuration and model storage controls.

| Area | Implemented behavior |
| --- | --- |
| Network editor | Ordered layer cards with add, remove, and move-up/move-down buttons |
| Presets | Dense feed-forward and convolutional MNIST classifiers |
| Editable layers | Dense, convolution, 2×2 max pooling, and dropout |
| Hidden activations | ReLU, tanh, sigmoid, and linear |
| Architecture inspection | Actual model output shapes and total parameter count |
| Help | Information buttons explaining editable settings and storage controls |
| Input validation | Immediate errors for invalid numeric drafts; training disabled until corrected |
| Training | Browser-based Adam optimization with configurable sample count, epochs, batch size, and learning rate |
| Comparison | Epoch-level loss and accuracy charts and a table for the last five runs |
| Evaluation | Accuracy on a fixed 1,000-image subset of the official test set |
| Prediction | Drawing canvas, random MNIST examples, predicted digit, and ten class scores |
| Persistence | Named browser-local architecture saves and trained-model saves |
| Portability | Import/export of architecture configuration JSON |
| Deployment | Static production build and GitHub Actions Pages workflow |

The editor builds sequential networks: each layer receives the previous layer's output. It is not a free-form graph editor. The app shows architecture and learning curves; it does not animate individual neurons, weights, gradients, or the steps of backpropagation.

## Run locally

### Prerequisites and checkout

Use Git and Node.js 22, which is also the version selected by the included CI workflow. npm is included with Node.js. Install dependencies from the lockfile with `npm ci`.

The project owner's authoritative Windows checkout is:

```text
C:\Users\Ruben\Documents\Personal\Neural Network Simulator\repo
```

For that existing checkout, open PowerShell and run:

```powershell
Set-Location 'C:\Users\Ruben\Documents\Personal\Neural Network Simulator\repo'
npm.cmd ci
npm.cmd run dev
```

For a new checkout on another machine:

```sh
git clone https://github.com/rubenlreyesr/neural-network-simulator.git
cd neural-network-simulator
npm ci
npm run dev
```

Open the HTTP address printed by Vite. The development server binds to `127.0.0.1`. Keep the terminal running while using the app; press **Ctrl+C** in that terminal to stop it.

On Windows, use `npm.cmd` if PowerShell blocks `npm.ps1`. If npm has been configured to use an unavailable script shell, a Windows command can explicitly select it, for example `npm.cmd run build --script-shell=cmd.exe`.

### Commands and production preview

| Command | Purpose |
| --- | --- |
| `npm ci` | Install exactly the dependency versions in `package-lock.json` |
| `npm run dev` | Start the Vite development server |
| `npm test` | Run the Node test suite |
| `npm run build` | Generate the static production site in `dist/` |
| `npm run preview` | Serve the existing production build locally |

To check the production build:

```sh
npm test
npm run build
npm run preview
```

Open the address printed by the preview server, commonly `http://127.0.0.1:4173/`. If that port is unavailable, use the address actually printed. Rebuild after source changes before checking preview again.

**Do not double-click `dist/index.html`.** Opening it as a `file://` URL can produce a blank page because the application uses JavaScript modules and fetches dataset files. Serve `dist/` over HTTP or HTTPS, even when running locally. The preview server is a local convenience; GitHub Pages serves the production files without a Node server.

## Your first experiment

1. Select the dense preset. Its hidden layer has 64 ReLU neurons.
2. Choose 1,000 training images for a quick trial. Leave epochs at 3, batch size at 64, and learning rate at 0.001.
3. Start training. The first run downloads and prepares the bundled MNIST data before epoch results appear.
4. Watch loss and accuracy. Dashed lines show training results; solid lines show validation results.
5. Evaluate the active model on the test sample when you want a separate held-out measurement.
6. Draw one large digit and select **Recognize digit**, or use **Use MNIST sample** to try an original dataset image.
7. Save the configuration **with weights** if you want to keep this trained model.
8. Select the convolution preset and train with the same settings. Compare the runs in the charts and table.

Begin with a small run rather than assuming a training duration. Time depends on the browser, available acceleration, model, sample count, and batch size. Keep the page open and preferably visible during training. Reloading loses unsaved weights and comparison history.

## Designing a network

### Input and automatic layers

Every model receives one grayscale **28×28** image, represented as a tensor with shape `[28, 28, 1]`. The final `1` is the number of image channels.

The application inserts a **Flatten** layer before the first dense layer, or before the output if necessary. Flatten changes a spatial tensor into a vector; it does not learn parameters. A raw 28×28 image becomes 784 values.

Every network ends with an automatic **10-neuron softmax output layer**, one output for each digit from 0 through 9. The output layer is fixed rather than editable because this application is specifically an MNIST classifier.

### Dense layers and neurons

A dense layer connects every input value to every neuron in that layer. Each neuron learns a weight for each input and a bias, then applies its activation function.

The **neurons** setting controls the width of the layer. More neurons increase capacity, parameter count, memory use, and computation. They do not automatically improve validation accuracy, particularly with a small dataset.

Allowed width: **1–256 neurons per editable dense layer**.

### Convolution layers: filters and kernels

A convolution layer applies small learned patterns across the image or incoming feature maps, preserving spatial structure.

**Filters** means the number of distinct patterns the layer learns. Each filter produces one output feature map, so a layer with 8 filters produces 8 output channels. During training, these patterns may become useful for detecting strokes, edges, curves, or combinations of earlier features; the app does not assign them predefined meanings.

**Kernel** means the spatial size of each sliding pattern. A kernel of 3 covers a **3×3** neighborhood; a kernel of 5 covers a **5×5** neighborhood. Each filter also spans all incoming channels. Larger kernels see a wider neighborhood and require more weights for the same channel count.

Allowed settings: **1–64 filters** and **3×3 or 5×5 kernels**. Convolutions use stride 1 and `same` padding, preserving image width and height. These stride and padding settings are fixed in the MVP.

### Max pooling

A max-pooling layer keeps the largest value from each 2×2 region, using a stride of 2. It reduces spatial dimensions and subsequent computation without learning weights. In the convolution preset, 28×28 becomes 14×14, then 7×7 after a second pool.

Pooling must operate on spatial data, before a dense layer, and the incoming width and height must each be at least 2. Pool size and stride are not configurable.

### Dropout

Dropout randomly suppresses a fraction of activations during training. A rate of `0.2` means approximately 20% are dropped on each application. This can discourage dependence on particular activations, but too much dropout can make learning difficult.

Allowed rate: **0–0.8**. Dropout adds no learned parameters and is inactive during ordinary inference. It can be placed among spatial or dense layers.

### Activation functions

| Activation | Behavior | Practical interpretation |
| --- | --- | --- |
| ReLU | Returns zero for negative inputs and the input for positive values | The default hidden activation in both presets |
| tanh | Maps values between -1 and 1 | Bounded and centered around zero; can saturate |
| sigmoid | Maps values between 0 and 1 | Bounded; can saturate and produce small gradients |
| linear | Leaves values unchanged | Adds no nonlinearity of its own |

Changing hidden activations does not change the final softmax layer. Stacking only linear dense layers does not provide the nonlinear representational benefit of typical multilayer networks.

### Preset architectures

The **dense preset** has **50,890 parameters**:

```text
Input 28×28×1
  → Flatten 784
  → Dense 64, ReLU
  → Dense 10, softmax
```

The **convolution preset** has **26,698 parameters**:

```text
Input 28×28×1
  → Conv 8 filters, 3×3, ReLU       28×28×8
  → Max pool 2×2                  14×14×8
  → Conv 16 filters, 3×3, ReLU     14×14×16
  → Max pool 2×2                   7×7×16
  → Flatten                       784
  → Dense 32, ReLU
  → Dense 10, softmax
```

Fewer parameters do not necessarily mean faster training. Convolution reuses each filter across many positions, so parameter count and operation count measure different things.

### Parameter counts and architecture limits

Parameters are the weights and biases learned during training. For these layers:

- Dense parameters: `(number of inputs + 1) × number of neurons`.
- Convolution parameters: `(kernel width × kernel height × incoming channels + 1) × filters`.
- Flatten, pooling, and dropout: zero learned parameters.

For example, the first dense preset layer has `(784 + 1) × 64 = 50,240` parameters, and its output has `(64 + 1) × 10 = 650`.

The editor supports at most **10 editable layers** and **2,000,000 total parameters**, including the automatic output. Convolution and pooling cannot follow a dense layer. An empty editable layer list is valid: the app can flatten the input and connect it directly to the output.

Architecture edits discard the current in-memory model and its weights. Save a trained model before editing if you want to return to it. Previously saved models and session comparison records are separate from the active model.

### Tooltips and numeric validation

Information buttons beside settings explain their meaning, examples, and tradeoffs. Help opens with hover, keyboard focus, or click/tap; **Escape** dismisses it. Pooling and automatic layers also have explanations.

Numeric fields allow you to type an incomplete or invalid draft so you can edit naturally. They do **not** silently clamp or round your input. For example, entering 31 epochs displays an error and prevents training; it does not run 31 epochs or silently change the value to 30. Empty values, non-finite values, and fractional values in integer-only settings are rejected. Correct the highlighted field to re-enable training.

## Training settings

| Setting | Default | Available values | Meaning |
| --- | --- | --- | --- |
| Training samples | 5,000 | 1,000 / 5,000 / 10,000 | Number of training images used in each epoch |
| Epochs | 3 | Integers 1–30 | Requested complete passes through the training subset |
| Batch size | 64 | 32 / 64 / 128 | Images processed together for one optimizer update |
| Learning rate | 0.001 | 0.01 / 0.001 / 0.0001 | Step-size setting for the Adam optimizer |

### Epochs

One epoch means one pass through the selected training images. With 5,000 samples and 3 epochs, the model processes those training examples three times, with shuffled training order. It does not see 15,000 unique images.

More epochs give the optimizer more opportunities to learn, but can eventually improve training performance while hurting validation performance. The app does not automatically stop at the best validation epoch or restore its weights.

### Batch size

The model computes predictions and gradients for a batch before updating its weights. With 1,000 examples and batch size 64, a complete epoch has 16 batches: fifteen batches of 64 and one of 40.

Larger batches typically use more memory and produce fewer updates per epoch. Smaller batches produce more frequent updates. The fastest or most effective choice depends on hardware and the model; the app does not automatically tune it.

### Learning rate

Learning rate influences how large the optimizer's updates are. A high value can learn quickly but may overshoot useful solutions or become unstable. A low value can learn slowly within the limited epoch budget. Adam adapts updates per parameter, so this is not a literal identical change applied to every weight.

Start with 0.001, then change one setting at a time. Adam and categorical cross-entropy are fixed choices in this MVP; there is no optimizer or loss selector.

### What happens during a run

Training prepares the dataset, creates a **fresh model**, compiles it, and fits it using shuffled batches. Validation uses a separate fixed 1,000-image subset after each epoch. Validation examples do not update the weights.

**Every press of Train starts from new weights**, including after loading a saved trained model. Loading weights is useful for inference and evaluation, not for resuming training in the current implementation.

During training, the display shows the current epoch, completed batches, and images processed in that epoch. The progress bar measures the current epoch and resets when the next epoch begins. Counters advance at batch boundaries, including the smaller final batch, with display updates throttled to about ten per second. A separate **Validating…** phase appears while the fixed validation subset is evaluated; a full training bar does not mean validation has finished. Stopped runs retain their actual partial-epoch image count.

Loss and accuracy charts still update when epoch results arrive, rather than after every batch. The reported elapsed time begins at the training request and can include data loading and preparation, so the first run's time is not a clean compute-speed comparison with later cached runs.

Stop is checked after a batch. It does not interrupt an ongoing data download or an individual computation immediately. A stopped run retains the active model's learned weights; the final reported epoch can be partial. If training produces non-finite metrics, the app reports the failure and disposes of the invalid model.

## Understanding results

**Accuracy** is the fraction of examples whose highest-scoring class matches the label. **Loss** is categorical cross-entropy: it reflects the probability assigned to the correct class, not just whether that class ranked first. Lower loss and higher accuracy are generally desirable, but they describe different aspects of performance.

Training metrics measure the data used for learning. Validation metrics measure a held-out subset used to assess generalization while experimenting. The charts use dashed lines for training and solid lines for validation, with colors distinguishing runs.

Useful patterns to look for:

- Both losses fall and validation accuracy rises: learning is helping on held-out examples.
- Training improves while validation deteriorates: possible overfitting.
- Both remain poor: try inspecting architecture, learning rate, or the training budget.
- Results fluctuate between runs: initialization and batch order are random; repeat comparable experiments.

The table retains the **last five runs in the current page session**, including architecture, settings, parameter count, elapsed time, validation accuracy, and test accuracy when recorded. Compact architecture labels use forms such as `D64` for a dense layer and `C8` for a convolution layer.

Only one model is active in memory. Old comparison rows are records, not selectable checkpoints holding their original weights. Save a model explicitly before replacing it if you want to use it again. Reloading the page clears comparison history, and loading a saved model does not reconstruct its training curves.

Test evaluation uses a fixed 1,000-image sample from the separate official MNIST test set. Evaluation of the current training run can attach its score to that row. A loaded model has no restored run record, so its evaluation result is reported without recreating a historical row.

Keep sample count, epochs, batch size, and learning rate equal for an initial architecture comparison. Repeat runs before drawing strong conclusions. Use validation for tuning and reserve test results for less frequent checks; repeatedly selecting designs based on test scores weakens the independence of that assessment.

## Drawing and prediction

Draw one digit using a mouse, pen, or touch pointer, then select **Recognize digit**. Prediction requires an active trained or loaded model. The output shows the selected class and all ten softmax scores.

The 280×280 drawing canvas is converted to model input by:

1. Finding the nonempty stroke bounds using an intensity threshold.
2. Cropping and scaling the digit to fit within a 20×20 region while preserving its aspect ratio.
3. Placing it on a 28×28 canvas.
4. Shifting it toward the center using its intensity centroid.
5. Scaling pixel values to the range 0–1.

A blank canvas produces an error. Draw one clear, reasonably large digit. The app does not deskew handwriting, apply data augmentation, or accept uploaded image files.

**Use MNIST sample** loads a random original test image and displays its true label. This is available once the dataset has loaded. Original samples use their original pixels for inference rather than passing through drawing normalization. These examples are selected from the full test set and can overlap the fixed evaluation subset.

Your handwriting can differ from MNIST in style, stroke width, and position. A model that performs well on MNIST can still misread a personal drawing. Softmax scores sum to approximately one, but a high score is not a calibrated guarantee that the prediction is correct.

## Saving and loading

| Operation | What it preserves | What it does not preserve |
| --- | --- | --- |
| Save configuration | Editable architecture and name | Learned weights and experiment history |
| Save with weights | Architecture, model topology, and learned weights | Charts, run history, and a resumable training session |
| Load configuration | Restores the editor architecture | A trained model unless weights were saved |
| Load saved model | Restores architecture and weights for inference/evaluation | Earlier run records or continued-training behavior |
| Export configuration | A downloadable architecture JSON file | Trained weights |
| Import configuration | Validated architecture from JSON | Weights or metrics |
| Delete saved item | Removes that stored entry and associated stored weights | Does not delete another separately saved item |

Names are limited to 60 characters; an empty name receives a generated name. Duplicate display names are allowed because entries have distinct identifiers. Deleting a stored entry asks for confirmation. It does not itself erase an already loaded in-memory model.

Configurations and the save index use `localStorage`, under `neural-lab-saves-v1`. Trained models use TensorFlow.js IndexedDB storage with identifiers beginning `indexeddb://neural-lab-`. Model metadata includes the editor configuration. If writing the save index fails after a model write, the app attempts to remove that newly stored model.

Browser saves belong to the **origin and browser profile**. The origin includes protocol, hostname, and port:

- `http://127.0.0.1:5173` and `http://127.0.0.1:4173` have different saves.
- `localhost` and `127.0.0.1` have different saves.
- Local development and the GitHub Pages site have different saves.
- Repository paths under the same `username.github.io` origin share browser storage and this application's save index.

Clearing site data can delete both configurations and weights. Private browsing, disabled storage, or exhausted quota can prevent saving. There is no account, cloud synchronization, or cross-device model transfer. Configuration export is useful for preserving a design, but **it is not a backup of a trained model**.

## Configuration JSON

Exports contain a version and an ordered list of editable layers. Automatic flattening and the final softmax layer are not included in that list. Training settings and weights are not part of the format.

A dense preset configuration:

```json
{
  "version": 1,
  "layers": [
    { "type": "dense", "units": 64, "activation": "relu" }
  ]
}
```

A convolutional design with optional dropout:

```json
{
  "version": 1,
  "layers": [
    { "type": "conv", "filters": 8, "kernel": 3, "activation": "relu" },
    { "type": "pool" },
    { "type": "conv", "filters": 16, "kernel": 3, "activation": "relu" },
    { "type": "pool" },
    { "type": "dense", "units": 32, "activation": "relu" },
    { "type": "dropout", "rate": 0.2 }
  ]
}
```

| Layer type | Fields | Constraints |
| --- | --- | --- |
| `dense` | `units`, `activation` | Integer units 1–256; supported activation |
| `conv` | `filters`, `kernel`, `activation` | Integer filters 1–64; kernel size 3 or 5; supported activation |
| `pool` | No additional editable fields | Must receive spatial data with dimensions at least 2×2 |
| `dropout` | `rate` | Finite number from 0 through 0.8 |

Supported activation strings are `relu`, `tanh`, `sigmoid`, and `linear`. Use JSON numbers for numeric fields, not numeric strings. Imported files must be no larger than **50,000 bytes**, use version 1, and pass architecture and resource validation. The export filename is `neural-lab-config.json`.

Importing changes the architecture and discards active weights. This is an application-specific editor format, not an arbitrary TensorFlow.js model importer.

## MNIST data and experiment design

The repository includes four original compressed MNIST IDX files in [`public/mnist/`](public/mnist/):

| File | Contents |
| --- | --- |
| `train-images-idx3-ubyte.gz` | 60,000 training images |
| `train-labels-idx1-ubyte.gz` | Their labels |
| `t10k-images-idx3-ubyte.gz` | 10,000 official test images |
| `t10k-labels-idx1-ubyte.gz` | Their labels |

Together they are approximately 12 MB compressed. Provenance and checksums are recorded in [`SOURCE.md`](public/mnist/SOURCE.md). The files came from the Google-hosted MNIST mirror; the running app fetches its bundled copies from its own static host.

The loader fetches all four files, even for a 1,000-image experiment. It detects gzip bytes and decompresses them when necessary, allowing for hosts that already decompress the response. It validates IDX magic values, 28×28 dimensions, and expected byte lengths. Each fetch has a 120-second timeout.

Pixels become floating-point values in `[0, 1]`, with input tensors shaped `[sample count, 28, 28, 1]`. Labels become ten-class one-hot vectors.

### Fixed subsets

Indices are deterministically shuffled using seed 42:

| Purpose | Selection |
| --- | --- |
| Training | First 1,000, 5,000, or 10,000 shuffled training indices |
| Validation | Shuffled training positions 50,000 through 50,999 |
| Test evaluation | First 1,000 separately shuffled official test indices |

Training and validation do not overlap. The smaller training subsets are contained within the larger ones. Sampling is not explicitly stratified by digit class.

This is a **subset experiment**, not a full-MNIST benchmark. The full files are present, but the interface does not train on all 60,000 examples or evaluate all 10,000 test examples. Deterministic subset selection does not make training fully reproducible: model initialization and training shuffle remain random.

Decoded data is cached in page memory. Failed loading can be retried. Normal browser HTTP caching may help later requests, but there is no service worker or managed offline dataset cache.

## GPU, NPU, browser support, and privacy

### Where computation runs

TensorFlow.js chooses an available backend, and the page header displays its name. Here, “backend” means a browser computation implementation, not a remote application server.

- **WEBGL:** TensorFlow.js uses the browser's WebGL path, which normally benefits from the local GPU. The browser may use software rendering, so the label alone does not prove which physical GPU is active.
- **CPU:** Computation uses the CPU backend and may be considerably slower for training.
- **NPU:** The application has no explicit NPU integration and does not claim to use an NPU.

There is no GPU picker or explicitly configured WebGPU, WebNN, or WASM backend. Data preparation, UI work, and other operations still use the CPU even when tensor computation uses WebGL. Training is not sent to a cloud GPU.

### Browser requirements

Use a modern browser with JavaScript modules, Fetch, typed arrays, canvas, pointer events, localStorage, IndexedDB, and `crypto.randomUUID`. Raw gzip responses require `DecompressionStream`. HTTPS hosting or localhost supports the secure-context APIs used by the app. WebGL availability affects acceleration.

The recorded interactive checks used Chromium. That is not a claim of complete Firefox, Safari, mobile, touch-device, or assistive-technology coverage. The interface includes labels, focusable help, keyboard-operable editor buttons, inline errors, and live status announcements, but has not undergone a comprehensive accessibility audit.

### Network and privacy boundaries

The static host supplies application assets and MNIST files. TensorFlow.js is bundled into the app. The stylesheet also requests optional Google Fonts, with local fallbacks.

Drawings, predictions, and trained weights are processed locally rather than uploaded to a training service. Saved models remain in browser storage. Ordinary asset and font requests still disclose normal connection metadata to their hosts. There is no application account system, analytics integration, or cloud save service in this MVP.

## GitHub Pages deployment

### Source and build artifacts

**Do not commit `dist/` for the included deployment method.** The repository tracks source, bundled data, the lockfile, and the workflow. GitHub Actions generates `dist/`, uploads it as a Pages artifact, and deploys that artifact. Both `dist/` and `node_modules/` are ignored by Git.

The existing workflow is [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml), named **Deploy to GitHub Pages**. It runs on pushes to `main` and can also be started manually.

### Repository setup and manual deployment

1. Open the repository's **Settings → Pages**.
2. Under **Build and deployment**, select **GitHub Actions** as the source.
3. Return to the repository's **top-level Actions tab**. This is separate from the **Actions** settings category in the Settings sidebar.
4. Select **Deploy to GitHub Pages** in the workflow list.
5. To deploy manually, use **Run workflow**, select `main`, and start it. A normal source push to `main` also triggers the workflow.
6. Open the run and inspect the `build` and `deploy` jobs. After deployment succeeds, follow its Pages URL.

The Jekyll and Static HTML cards in Pages settings are suggested workflow templates. You do not need to configure either for this repository because a build-and-deploy workflow already exists.

If the named workflow is absent, verify that `.github/workflows/deploy.yml` is committed on the repository's default branch. Check branch selection and repository Actions permissions before creating another workflow. If deploying from a branch other than `main`, update the workflow's push filter accordingly.

### Workflow behavior

The build job checks out the source, selects Node.js 22, installs locked dependencies, runs `npm test`, builds the app, configures Pages, and uploads `dist/`. The deployment job consumes that artifact and deploys it to the `github-pages` environment.

The workflow grants `contents: read`, `pages: write`, and `id-token: write`. Pages runs share a concurrency group, with newer runs allowed to cancel earlier in-progress runs. This setup requires no manually created `gh-pages` branch, production server, API keys, or application environment variables.

[`vite.config.js`](vite.config.js) sets `base: './'`, allowing relative assets to work under a repository subpath such as `/neural-network-simulator/`. The app has no client-side path routes requiring fallback rewrites. Vite copies `public/mnist/` into the production output.

GitHub Actions has previously reported Node.js runtime deprecation notices for action versions in this workflow. Those action-runtime notices are distinct from the Node.js version used to build the project. A warning alone is not a failed deployment: inspect the actual job conclusion and error output. Updating action versions is a separate maintenance task; this README does not change them.

For GitHub's platform guidance, see [using custom workflows with GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Repository guide

| Path | Responsibility |
| --- | --- |
| [`index.html`](index.html) | HTML entry point |
| [`src/main.js`](src/main.js) | UI state, layer editor, training lifecycle, charts, prediction, and saves |
| [`src/model.js`](src/model.js) | Presets, configuration validation, model construction, and disposal |
| [`src/data.js`](src/data.js) | Downloads, gzip/IDX decoding, deterministic subsets, and tensor preparation |
| [`src/drawing.js`](src/drawing.js) | Drawing canvas and digit preprocessing |
| [`src/help.js`](src/help.js) | Explanations used by information buttons |
| [`src/validation.js`](src/validation.js) | Numeric input validation |
| [`src/style.css`](src/style.css) | Layout, colors, typography, and responsive/help styles |
| [`tests/model.test.js`](tests/model.test.js) | Model, data, training, serialization, and cleanup checks |
| [`tests/validation.test.js`](tests/validation.test.js) | Numeric validation checks |
| [`public/mnist/SOURCE.md`](public/mnist/SOURCE.md) | Dataset source and checksums |
| [`package.json`](package.json) | Scripts and dependency declarations |
| [`package-lock.json`](package-lock.json) | Locked dependency tree |
| [`vite.config.js`](vite.config.js) | Relative production asset base |
| [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) | CI and Pages deployment |
| [`VERIFICATION.md`](VERIFICATION.md) | Historical implementation verification notes |
| [`docs/help-preview.png`](docs/help-preview.png) | Help-interface reference screenshot |

When adding a layer or editable setting, consider model validation, construction, UI controls, help text, import/export compatibility, and tests together. Release TensorFlow.js tensors and models when replacing them; JavaScript garbage collection alone is not a substitute for tensor disposal. The disposal helper also disposes of an attached optimizer.

This project uses plain JavaScript ES modules, not a TypeScript build or a UI framework. Most application coordination currently lives in `src/main.js`, appropriate to the modest MVP but a likely refactoring point if the feature set grows.

## Verification

Run `npm test` and `npm run build` before publishing implementation changes. The suite covers:

- Preset parameter counts, output shapes, and normalized prediction scores.
- Invalid configurations, layer ordering, numeric ranges, and resource limits.
- Deterministic subset selection and training/validation separation.
- IDX parsing and invalid input handling.
- Bundled dataset hashes, dimensions, counts, and label ranges.
- Actual small synthetic training runs for dense and convolutional models.
- Serialization/inference equivalence and tensor cleanup within the model tests.
- Numeric validation, including incomplete and out-of-range input.
- Batch progress, final partial batches, validation metric equivalence, and stopped-run validation.

These automated tests use the CPU backend and do not replace browser interaction checks or real-MNIST learning experiments. A successful build confirms that production assets can be generated; it does not by itself prove that a deployed URL works.

Historical browser verification included real MNIST training, WebGL inference, save/load, drawing recognition, subpath hosting, and the help/validation interactions. One early dense run using 1,000 training samples and 3 epochs reached 78.8% validation and 80.3% test accuracy; an early convolution run reached 55.9% validation accuracy. These are individual smoke-test outcomes, not expected scores, performance promises, or evidence that one architecture is generally superior.

[`VERIFICATION.md`](VERIFICATION.md) records checks at an earlier point in the project. Statements there about deployment status describe that historical checkpoint, not the current live site's status. Dependency/build warnings should be assessed separately from failed tests or deployment errors.

## Troubleshooting

| Symptom | Explanation or next step |
| --- | --- |
| Blank page after double-clicking `dist/index.html` | Use `npm run preview` and open its HTTP URL; do not use `file://`. |
| Preview does not show recent source changes | Run `npm run build` again before previewing. |
| PowerShell blocks npm | Use `npm.cmd` instead of the blocked `npm.ps1` wrapper. |
| npm references a missing shell | Inspect npm's script-shell configuration; on Windows, try `--script-shell=cmd.exe`. |
| Local URL does not respond | Keep the dev/preview process running and use the exact printed address and port. |
| First training run appears idle | Dataset download and preparation precede epoch-level chart updates. Inspect status and network failures. |
| Dataset request fails | Check connection and that all four files are served under the site's `mnist/` directory; retry after correcting the failure. |
| Decompression API error | Use a browser supporting `DecompressionStream` for gzip responses. |
| Epoch field accepts 31 while typing | Draft text is editable, but validation rejects it and disables training. Enter an integer from 1 through 30. |
| Training button remains disabled | Correct inline validation errors and invalid layer order, or wait for the active operation to finish. |
| Convolution/pooling cannot follow dense | Move spatial layers before the first dense layer. |
| Architecture exceeds limits | Reduce layers, neurons, or filters; the cap includes output-layer parameters. |
| Stop does not react instantly | It takes effect at a batch boundary and does not abort dataset downloading. |
| Training is slow | Try 1,000 samples, fewer epochs, or a smaller network; check the reported computation backend. |
| High MNIST accuracy but poor drawings | Handwriting differs from the dataset; try a clear centered digit and compare with an original MNIST example. |
| High score on an incorrect digit | Softmax confidence is not calibrated correctness. |
| Saved models appear missing | Check browser profile, protocol, hostname, and port; saves are origin-specific. |
| Save fails | Browser storage may be restricted or full; inspect the reported error. |
| Exported JSON does not recover learned behavior | Configuration export contains architecture only, not weights. |
| Training a loaded model changes its predictions | Train intentionally creates fresh weights; resume is not implemented. |
| Old runs disappear after reload | Comparison history is session-only. |
| No Deploy to GitHub Pages entry in Settings | Open the repository's top-level Actions tab; Settings → Actions controls permissions. |
| Pages offers Jekyll or Static HTML | Those are optional templates; use the repository's existing deployment workflow. |
| Pages is not updated after a push | Check the pushed branch and latest Actions run, including both build and deploy jobs. |
| Actions shows a Node runtime notice | Distinguish warnings from failed job conclusions and inspect the failing step if any. |
| Production build reports a large bundle | TensorFlow.js contributes substantial JavaScript; a size warning alone does not mean the build failed. |

## Future enhancements — not implemented

The following are possible directions, not shipped features or commitments:

- Drag-and-drop graph editing, branching networks, and custom connections.
- Individual neuron, weight, activation, gradient, and backpropagation visualizations.
- Other datasets, image uploads, full-MNIST training, and full test-set evaluation.
- Additional optimizers, losses, seeds, learning-rate schedules, and early stopping.
- Continued training from saved weights and best-epoch checkpoints.
- Confusion matrices, per-class metrics, and richer experiment comparison.
- Persistent experiment history and selectable model checkpoints for old runs.
- Downloadable trained-model backups and arbitrary TensorFlow.js model import.
- Web Workers, managed offline caching, and dedicated mobile performance work.
- Explicit WebGPU, WebNN/NPU integration, or hardware selection.
- Comprehensive browser, touch-device, and accessibility verification.

## References and attribution

- [Bundled MNIST provenance and checksums](public/mnist/SOURCE.md)
- [TensorFlow.js API reference](https://js.tensorflow.org/api/latest/)
- [TensorFlow.js model save/load guide](https://www.tensorflow.org/js/guide/save_load)
- [GitHub custom Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)

Third-party dependencies and dataset materials retain their respective terms. Consult their distributions and source documentation when redistributing the project or its assets.
