import * as tf from "@tensorflow/tfjs";
import { presets, validate, buildModel, disposeModel } from "./model.js";
import { loadData, indices, tensors } from "./data.js";
import { normalizeDigit } from "./drawing.js";
import "./style.css";
import { attachHelp } from "./help.js";
import { numericError } from "./validation.js";
const $ = (s) => document.querySelector(s),
  clone = (x) => structuredClone(x);
let config = { version: 1, layers: clone(presets.dense) },
  model = null,
  busy = false,
  ready = false,
  trained = false,
  stop = false,
  data,
  runs = [];
const colors = ["#268568", "#c66d3a", "#677bc1", "#ad5892", "#8b842d"];
document.querySelector("#app").innerHTML = `
<header><a class="brand" href="#">◈ <span>NEURAL LAB</span></a><span class="badge">MNIST PLAYGROUND / MVP 0.1</span><span id="backend">Initializing…</span></header>
<main><section class="intro"><div><p class="eyebrow">A LITTLE NETWORK. A LOT TO DISCOVER.</p><h1>Build it. Train it.<br><em>See what it learns.</em></h1><p>Turn handwritten digits into your first neural network experiment.<br>Everything trains right here, in your browser.</p></div><div class="recipe"><b>YOUR FIRST EXPERIMENT</b><span>01 &nbsp; Start with the dense preset</span><span>02 &nbsp; Train for 3 epochs</span><span>03 &nbsp; Draw a digit. Try a convolution next.</span></div></section>
<div id="status" role="status" aria-live="polite">Preparing TensorFlow.js…</div>
<div class="workspace"><section class="panel editor"><div class="section-title"><h2><span>01</span> Design</h2><span class="muted">28 × 28 → 10 classes</span></div>
<fieldset id="design"><div class="row"><button id="dense">Dense preset</button><button id="cnn">Convolution preset</button></div><p class="hint">Layers run from top to bottom. Flatten and the 10-class softmax output are added automatically.</p><div id="layers"></div><div class="row"><select id="add-type" aria-label="New layer type"><option value="dense">Dense</option><option value="conv">Convolution 2D</option><option value="pool">Max pooling 2D</option><option value="dropout">Dropout</option></select><button id="add">＋ Add layer</button></div></fieldset>
<div id="architecture"></div><details><summary>Configuration & saved models</summary><div class="storage"><div class="row"><button id="export">Export config</button><label class="file-button">Import config<input id="import" type="file" accept="application/json,.json"></label></div><label>Save name<input id="save-name" maxlength="60" placeholder="My experiment"></label><div class="row"><button id="save-config">Save config</button><button id="save-model">Save weights + config</button></div><label>Browser saves<select id="saved"><option value="">No saves yet</option></select></label><div class="row"><button id="load">Load selected</button><button id="delete">Delete selected</button></div><p class="hint">Saved on this browser and origin only. Clearing site data removes saves. JSON exports contain configuration only.</p></div></details></section>
<section class="panel training"><div class="section-title"><h2><span>02</span> Train & compare</h2><span class="badge">REAL MNIST DATA</span></div><fieldset id="settings"><div class="settings"><label>Training images<select id="samples"><option value="1000">1,000 · quick</option><option value="5000" selected>5,000 · starter</option><option value="10000">10,000</option></select></label><label>Epochs<input id="epochs" type="number" min="1" max="30" value="3"></label><label>Batch size<select id="batch"><option>32</option><option selected>64</option><option>128</option></select></label><label>Learning rate<select id="lr"><option>0.01</option><option selected>0.001</option><option>0.0001</option></select></label></div></fieldset>
<p id="validation-message" class="validation-message" role="status" aria-live="polite"></p><div class="row actions"><button id="train" class="primary" disabled>▶ Train new run</button><button id="stop" disabled>Stop</button><button id="evaluate" disabled>Evaluate test set</button></div><p class="hint">Each run starts fresh. Adam optimizer · fixed 1,000-image validation set · test set stays separate. Training may take seconds to minutes.</p><progress id="progress" max="1" value="0" aria-label="Training progress"></progress><div class="metrics"><div><small>VALIDATION ACCURACY</small><strong id="accuracy">—</strong></div><div><small>VALIDATION LOSS</small><strong id="loss">—</strong></div><div><small>EPOCH</small><strong id="epoch">0 / 3</strong></div></div>
<div class="chart-title"><h3>Accuracy</h3><span>solid: validation · dashed: training</span></div><div id="accuracy-chart" class="chart"></div><div class="chart-title"><h3>Loss</h3><span>lower is better</span></div><div id="loss-chart" class="chart"></div><div id="legend"></div><div class="table-scroll"><table><thead><tr><th>Run / architecture</th><th>Images · LR · batch</th><th>Val. acc.</th><th>Test acc.</th></tr></thead><tbody id="runs"><tr><td colspan="4">Your experiments will appear here.</td></tr></tbody></table></div><p class="hint">Last five runs shown; history lasts for this page session. Initialization and batch order are random, so results vary.</p></section>
<section class="panel inference"><div class="section-title"><h2><span>03</span> Try a digit</h2><span id="model-state" class="muted">Not trained</span></div><p class="hint">Draw one large digit, 0–9. Train a model first.</p><div class="drawing-area"><canvas id="draw" width="280" height="280" aria-label="Draw a digit using a mouse or touch"></canvas><div class="predictions"><div class="prediction"><span id="prediction">?</span><p id="confidence">Waiting for a trained model</p></div><div id="bars"></div></div></div><div class="row"><button id="clear">Clear canvas</button><button id="predict" class="primary" disabled>Recognize digit</button><button id="example" disabled>Use MNIST sample</button></div><p id="sample-label" class="hint">Drawings are cropped, scaled, and centered before prediction. Confidence is a model score, not a guarantee.</p></section></div>
<footer><span>Small experiments, real learning.</span><span>TensorFlow.js · No account · No backend</span></footer></main>`;
function status(message, error = false) {
  $("#status").textContent = message;
  $("#status").classList.toggle("error", error);
}
function guard(fn) {
  return async (...args) => {
    try {
      await fn(...args);
    } catch (e) {
      status(e.message || String(e), true);
      console.error(e);
    }
  };
}
function lock(value, stoppable = false) {
  busy = value;
  $("#design").disabled = value;
  $("#settings").disabled = value;
  for (const id of [
    "train",
    "load",
    "delete",
    "save-config",
    "save-model",
    "export",
    "import",
  ])
    $("#" + id).disabled = value;
  $("#stop").disabled = !value || !stoppable;
  $("#evaluate").disabled = value || !trained;
  $("#predict").disabled = value || !trained;
  $("#example").disabled = value || !data;
  validateControls();
}
function resetModel() {
  disposeModel(model);
  model = null;
  trained = false;
  $("#model-state").textContent = "Not trained";
  $("#prediction").textContent = "?";
  $("#confidence").textContent = "Train this architecture to predict";
  $("#bars").innerHTML = "";
  $("#accuracy").textContent = "—";
  $("#loss").textContent = "—";
  $("#epoch").textContent = "0 / " + $("#epochs").value;
  $("#progress").value = 0;
  lock(false);
}
function options(values, selected) {
  return values
    .map((v) => `<option ${v === selected ? "selected" : ""}>${v}</option>`)
    .join("");
}
function renderEditor() {
  $("#layers").innerHTML = config.layers
    .map(
      (l, i) =>
        `<div class="layer"><div class="layer-top"><b>${String(i + 1).padStart(2, "0")} &nbsp; ${{ dense: "Dense", conv: "Convolution 2D", pool: "Max pooling 2D", dropout: "Dropout" }[l.type]}</b><div><button data-action="up" data-index="${i}" aria-label="Move layer ${i + 1} up" ${i === 0 ? "disabled" : ""}>↑</button><button data-action="down" data-index="${i}" aria-label="Move layer ${i + 1} down" ${i === config.layers.length - 1 ? "disabled" : ""}>↓</button><button data-action="remove" data-index="${i}" aria-label="Remove layer ${i + 1}">×</button></div></div><div class="layer-fields">${l.type === "dense" ? `<label>Neurons<input data-index="${i}" data-key="units" type="number" min="1" max="256" value="${l.units}"></label>` : ""}${l.type === "conv" ? `<label>Filters<input data-index="${i}" data-key="filters" type="number" min="1" max="64" value="${l.filters}"></label><label>Kernel<select data-index="${i}" data-key="kernel">${options([3, 5], l.kernel)}</select></label>` : ""}${["dense", "conv"].includes(l.type) ? `<label>Activation<select data-index="${i}" data-key="activation">${options(["relu", "tanh", "sigmoid", "linear"], l.activation)}</select></label>` : ""}${l.type === "dropout" ? `<label>Drop rate<input data-index="${i}" data-key="rate" type="number" min="0" max="0.8" step="any" value="${l.rate}"></label>` : ""}${l.type === "pool" ? '<span class="hint" data-help="pool">2 × 2 window · stride 2</span>' : ""}</div></div>`,
    )
    .join("");
  attachHelp($("#layers"));
  renderArchitecture();
}
function renderArchitecture() {
  let temp;
  try {
    const count = validate(config);
    temp = buildModel(config);
    $("#architecture").innerHTML =
      `<div class="parameter"><b>${count.toLocaleString()}</b> trainable parameters</div><div class="flow"><span>Input <b>28 × 28 × 1</b></span>${temp.layers.map((l) => `<span ${l.getClassName() === "Flatten" ? 'data-help="flatten"' : l === temp.layers.at(-1) ? 'data-help="output"' : l.getClassName() === "MaxPooling2D" ? 'data-help="pool"' : ""}>↓ ${l.getClassName()} <b>${l.outputShape.slice(1).join(" × ")}</b></span>`).join("")}</div>`;
    attachHelp($("#architecture"));
  } catch (e) {
    $("#architecture").textContent = e.message;
    $("#train").disabled = true;
  } finally {
    disposeModel(temp);
    validateControls();
  }
}
function changeConfig(next) {
  resetModel();
  config = next;
  renderEditor();
  status("Architecture updated. The next run starts with new weights.");
}
// Update in place while typing so validation never steals focus or erases a draft.
$("#layers").addEventListener("input", (e) => {
  const { index, key } = e.target.dataset;
  if (!key) return;
  resetModel();
  config.layers[+index][key] =
    key === "activation"
      ? e.target.value
      : e.target.value === ""
        ? NaN
        : Number(e.target.value);
  renderArchitecture();
});
$("#settings").addEventListener("input", validateControls);
function validateControls() {
  const errors = [];
  for (const input of document.querySelectorAll('input[type="number"]')) {
    const message = numericError(input);
    input.setAttribute("aria-invalid", String(!!message));
    const id = input.id + "-error";
    let error = document.getElementById(id);
    if (!error) {
      error = document.createElement("span");
      error.id = id;
      error.className = "field-error";
      error.setAttribute("aria-live", "polite");
      input.after(error);
    }
    error.textContent = message;
    input.setAttribute("aria-describedby", id);
    if (message) errors.push(message);
  }
  try {
    validate(config);
  } catch (e) {
    if (!errors.length) errors.push(e.message);
  }
  $("#validation-message").textContent = errors.length
    ? "Correct the highlighted settings before training. " + errors.join(" ")
    : "";
  $("#train").disabled = busy || !ready || errors.length > 0;
  return errors.length === 0;
}
$("#layers").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b?.dataset.action) return;
  const next = clone(config),
    i = +b.dataset.index;
  if (b.dataset.action === "remove") next.layers.splice(i, 1);
  else {
    const j = i + (b.dataset.action === "up" ? -1 : 1);
    [next.layers[i], next.layers[j]] = [next.layers[j], next.layers[i]];
  }
  changeConfig(next);
});
for (const name of ["dense", "cnn"])
  $("#" + name).onclick = () =>
    changeConfig({ version: 1, layers: clone(presets[name]) });
$("#add").onclick = guard(() => {
  if (config.layers.length >= 10) throw Error("Maximum 10 editable layers.");
  const type = $("#add-type").value;
  changeConfig({
    version: 1,
    layers: [
      ...config.layers,
      clone(
        {
          dense: { type, units: 32, activation: "relu" },
          conv: { type, filters: 8, kernel: 3, activation: "relu" },
          pool: { type },
          dropout: { type, rate: 0.2 },
        }[type],
      ),
    ],
  });
});
function chart(kind) {
  const W = 580,
    H = 155,
    L = 38,
    R = 12,
    T = 12,
    B = 25,
    maxEpoch = Math.max(2, ...runs.map((r) => r.history.length));
  const all = runs.flatMap((r) =>
    r.history.flatMap((h) => [h[kind], h["val_" + kind]]),
  );
  const max =
    kind === "acc" ? 1 : Math.max(0.1, ...all.filter(Number.isFinite)) * 1.1;
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${kind === "acc" ? "Accuracy" : "Loss"} by epoch for recent runs">`;
  for (let n = 0; n <= 4; n++) {
    const y = T + ((H - T - B) * n) / 4;
    svg += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="#e4e9e2"/><text x="${L - 6}" y="${y + 4}" text-anchor="end">${(max * (1 - n / 4)).toFixed(kind === "acc" ? 2 : 1)}</text>`;
  }
  runs.forEach((r) => {
    for (const [key, dash] of [
      [kind, true],
      ["val_" + kind, false],
    ]) {
      const pts = r.history
        .map(
          (h, i) =>
            `${L + (i / (maxEpoch - 1)) * (W - L - R)},${T + (1 - h[key] / max) * (H - T - B)}`,
        )
        .join(" ");
      svg += `<polyline points="${pts}" fill="none" stroke="${r.color}" stroke-width="2" ${dash ? 'stroke-dasharray="5 4"' : ""}/>`;
      r.history.forEach((h, i) => {
        svg += `<circle cx="${L + (i / (maxEpoch - 1)) * (W - L - R)}" cy="${T + (1 - h[key] / max) * (H - T - B)}" r="2" fill="${r.color}"/>`;
      });
    }
  });
  svg += `<text x="${L}" y="${H - 4}">Epoch 1</text><text x="${W - R}" y="${H - 4}" text-anchor="end">${maxEpoch}</text></svg>`;
  return svg;
}
function renderRuns() {
  for (const [id, key] of [
    ["accuracy", "acc"],
    ["loss", "loss"],
  ])
    $("#" + id + "-chart").innerHTML = chart(key);
  $("#legend").innerHTML = runs
    .map(
      (r) => `<span style="color:${r.color}">● Run ${r.id} ${r.state}</span>`,
    )
    .join(" ");
  if (runs.length)
    $("#runs").innerHTML = runs
      .map(
        (r) =>
          `<tr><td><b style="color:${r.color}">Run ${r.id}</b> · ${r.config.layers.map((l) => (l.type === "dense" ? `D${l.units}` : l.type === "conv" ? `C${l.filters}` : l.type)).join(" → ") || "linear"}<br><small>${r.params.toLocaleString()} params · ${r.history.length} epochs${r.seconds ? ` · ${r.seconds}s` : ""}</small></td><td>${r.samples.toLocaleString()} · ${r.lr} · ${r.batch}</td><td>${r.history.length ? (r.history.at(-1).val_acc * 100).toFixed(1) + "%" : "—"}</td><td>${r.test == null ? "—" : (r.test * 100).toFixed(1) + "%"}</td></tr>`,
      )
      .join("");
}
let runId = 0,
  currentRun = null;
$("#train").onclick = guard(async () => {
  if (!validateControls())
    throw Error("Correct the highlighted settings before training.");
  validate(config);
  const epochs = Number($("#epochs").value);
  if (!Number.isInteger(epochs) || epochs < 1 || epochs > 30)
    throw Error("Choose 1–30 epochs.");
  resetModel();
  lock(true, true);
  stop = false;
  let sets = [];
  const started = performance.now();
  let run;
  try {
    status("Downloading MNIST (~12 MB compressed on first use)…");
    data = await loadData();
    if (stop) {
      status("Stopped before training.");
      return;
    }
    model = buildModel(config);
    model.compile({
      optimizer: tf.train.adam(Number($("#lr").value)),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });
    const order = indices(data.trainLabels.length),
      samples = Number($("#samples").value),
      batch = Number($("#batch").value);
    const train = tensors(
      data.trainImages,
      data.trainLabels,
      order.slice(0, samples),
    );
    sets.push(...train);
    const val = tensors(
      data.trainImages,
      data.trainLabels,
      order.slice(50000, 51000),
    );
    sets.push(...val);
    run = {
      id: ++runId,
      config: clone(config),
      params: model.countParams(),
      samples,
      batch,
      lr: Number($("#lr").value),
      history: [],
      color: colors[(runId - 1) % 5],
      state: "training",
    };
    currentRun = run;
    runs.push(run);
    runs = runs.slice(-5);
    renderRuns();
    $("#progress").value = 0;
    status(`Run ${run.id}: training epoch 1 of ${epochs}…`);
    await model.fit(...train, {
      epochs,
      batchSize: batch,
      validationData: val,
      shuffle: true,
      yieldEvery: "batch",
      callbacks: {
        onBatchEnd: async () => {
          if (stop) model.stopTraining = true;
        },
        onEpochEnd: async (epoch, logs) => {
          if (!Object.values(logs).every(Number.isFinite))
            throw Error("Training became unstable. Try a lower learning rate.");
          run.history.push({ ...logs });
          $("#accuracy").textContent = (logs.val_acc * 100).toFixed(1) + "%";
          $("#loss").textContent = logs.val_loss.toFixed(3);
          $("#epoch").textContent = `${epoch + 1} / ${epochs}`;
          $("#progress").value = (epoch + 1) / epochs;
          status(
            `Run ${run.id}: epoch ${epoch + 1} of ${epochs}${stop ? " · stopping" : ""}`,
          );
          renderRuns();
          await tf.nextFrame();
        },
      },
    });
    trained = true;
    run.state = stop ? "stopped" : "complete";
    run.seconds = Math.round((performance.now() - started) / 1000);
    $("#model-state").textContent = `Run ${run.id} · ${run.state}`;
    status(
      stop
        ? "Stopped. Current weights are available for prediction and saving."
        : "Training complete. Draw a digit, evaluate the test set, or change the architecture for a new run.",
    );
  } catch (e) {
    if (run) run.state = "failed";
    resetModel();
    throw e;
  } finally {
    tf.dispose(sets);
    lock(false);
    renderArchitecture();
    renderRuns();
  }
});
$("#stop").onclick = () => {
  stop = true;
  if (model) model.stopTraining = true;
  $("#stop").disabled = true;
  status("Stopping after the current batch or download completes…");
};
$("#evaluate").onclick = guard(async () => {
  lock(true);
  let set = [],
    result;
  try {
    status("Evaluating a fixed 1,000-image sample of the official test set…");
    data = await loadData();
    set = tensors(
      data.testImages,
      data.testLabels,
      indices(data.testLabels.length).slice(0, 1000),
    );
    result = model.evaluate(...set, { batchSize: 64 });
    const acc = (await result[1].data())[0];
    if (currentRun && runs.includes(currentRun)) currentRun.test = acc;
    renderRuns();
    status(
      `Held-out test accuracy: ${(acc * 100).toFixed(2)}% on 1,000 images. Repeated tuning on this result makes it less independent.`,
    );
  } finally {
    tf.dispose(set);
    tf.dispose(result);
    lock(false);
  }
});
const canvas = $("#draw"),
  ctx = canvas.getContext("2d");
let drawing = false,
  examplePixels = null;
function clear() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, 280, 280);
  examplePixels = null;
  $("#prediction").textContent = "?";
  $("#confidence").textContent = trained
    ? "Ready for your digit"
    : "Waiting for a trained model";
  $("#bars").innerHTML = "";
  $("#sample-label").textContent =
    "Drawings are cropped, scaled, and centered before prediction. Confidence is a model score, not a guarantee.";
}
function point(e) {
  const r = canvas.getBoundingClientRect();
  return [
    ((e.clientX - r.left) * 280) / r.width,
    ((e.clientY - r.top) * 280) / r.height,
  ];
}
canvas.onpointerdown = (e) => {
  drawing = true;
  examplePixels = null;
  canvas.setPointerCapture(e.pointerId);
  ctx.strokeStyle = "white";
  ctx.fillStyle = "white";
  ctx.lineWidth = 20;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const [x, y] = point(e);
  ctx.beginPath();
  ctx.arc(x, y, 10, 0, 2 * Math.PI);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(x, y);
};
canvas.onpointermove = (e) => {
  if (drawing) {
    ctx.lineTo(...point(e));
    ctx.stroke();
  }
};
canvas.onpointerup = canvas.onpointercancel = () => {
  drawing = false;
};
$("#clear").onclick = clear;
$("#predict").onclick = guard(async () => {
  const pixels = examplePixels || normalizeDigit(canvas);
  if (!pixels) throw Error("Draw a digit first.");
  const scores = tf.tidy(() =>
    model.predict(tf.tensor4d(pixels, [1, 28, 28, 1])).dataSync(),
  );
  const best = Array.from(scores).indexOf(Math.max(...scores));
  $("#prediction").textContent = best;
  $("#confidence").textContent =
    `${(scores[best] * 100).toFixed(1)}% model confidence`;
  $("#bars").innerHTML = Array.from(scores)
    .map(
      (p, i) =>
        `<div class="bar"><b>${i}</b><span><i style="width:${p * 100}%"></i></span><small>${(p * 100).toFixed(1)}%</small></div>`,
    )
    .join("");
});
$("#example").onclick = () => {
  clear();
  const id = Math.floor(Math.random() * data.testLabels.length);
  examplePixels = Float32Array.from(
    data.testImages.subarray(id * 784, (id + 1) * 784),
    (x) => x / 255,
  );
  const tiny = document.createElement("canvas");
  tiny.width = tiny.height = 28;
  const t = tiny.getContext("2d"),
    im = t.createImageData(28, 28);
  examplePixels.forEach((v, i) => {
    im.data[i * 4] = im.data[i * 4 + 1] = im.data[i * 4 + 2] = v * 255;
    im.data[i * 4 + 3] = 255;
  });
  t.putImageData(im, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tiny, 0, 0, 280, 280);
  $("#sample-label").textContent =
    `Official test image · true label ${data.testLabels[id]}. Used as-is, without drawing normalization.`;
};
const storageKey = "neural-lab-saves-v1";
function saved() {
  return JSON.parse(localStorage.getItem(storageKey) || "[]");
}
function refreshSaves() {
  const select = $("#saved");
  select.replaceChildren();
  for (const item of saved()) {
    const o = document.createElement("option");
    o.value = item.id;
    o.textContent = `${item.name} · ${item.weights ? "weights + config" : "config only"}`;
    select.append(o);
  }
}
async function save(weights) {
  validate(config);
  if (weights && !trained)
    throw Error("Train or load a model before saving weights.");
  lock(true);
  const id = crypto.randomUUID();
  try {
    const item = {
      id,
      name:
        $("#save-name").value.trim() ||
        `Experiment ${new Date().toLocaleString()}`,
      config: clone(config),
      weights,
    };
    if (weights) {
      model.setUserDefinedMetadata({ neuralLab: config });
      await model.save("indexeddb://neural-lab-" + id);
    }
    try {
      localStorage.setItem(storageKey, JSON.stringify([...saved(), item]));
    } catch (e) {
      if (weights) await tf.io.removeModel("indexeddb://neural-lab-" + id);
      throw e;
    }
    refreshSaves();
    $("#saved").value = id;
    status("Saved in this browser.");
  } finally {
    lock(false);
    renderArchitecture();
  }
}
$("#save-config").onclick = guard(() => save(false));
$("#save-model").onclick = guard(() => save(true));
$("#load").onclick = guard(async () => {
  const item = saved().find((x) => x.id === $("#saved").value);
  if (!item) throw Error("Select a saved experiment.");
  validate(item.config);
  lock(true);
  let loaded;
  try {
    if (item.weights)
      loaded = await tf.loadLayersModel("indexeddb://neural-lab-" + item.id);
    resetModel();
    config = clone(item.config);
    model = loaded || null;
    trained = !!loaded;
    currentRun = null;
    renderEditor();
    $("#model-state").textContent = trained
      ? "Saved model loaded"
      : "Not trained";
    status(`Loaded ${item.name}. Training again creates a fresh run.`);
  } finally {
    lock(false);
    renderArchitecture();
  }
});
$("#delete").onclick = guard(async () => {
  const items = saved(),
    item = items.find((x) => x.id === $("#saved").value);
  if (!item) throw Error("Select a save to delete.");
  if (!confirm(`Delete browser save “${item.name}”?`)) return;
  if (item.weights)
    await tf.io.removeModel("indexeddb://neural-lab-" + item.id);
  localStorage.setItem(
    storageKey,
    JSON.stringify(items.filter((x) => x.id !== item.id)),
  );
  refreshSaves();
  status("Browser save deleted.");
});
$("#export").onclick = guard(() => {
  validate(config);
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(config, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "neural-lab-config.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$("#import").onchange = guard(async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  if (file.size > 50000) throw Error("Configuration file is too large.");
  const parsed = JSON.parse(await file.text());
  validate(parsed);
  changeConfig(parsed);
});
attachHelp();
clear();
renderRuns();
async function initialize() {
  try {
    await tf.ready();
    ready = true;
    $("#backend").textContent = `● ${tf.getBackend().toUpperCase()} backend`;
    renderEditor();
    lock(false);
    status("Ready. Design a network, then start your first run.");
    try {
      refreshSaves();
    } catch {
      status(
        "Browser storage is unavailable. Training still works; use configuration export.",
      );
    }
  } catch (e) {
    status("TensorFlow.js could not initialize: " + e.message, true);
  }
}
initialize();
