// Help stays beside the control and works with pointer, keyboard, and touch.
const topics = {
  units: [
    "Neurons",
    "How many learned units this dense layer contains. Choose 1–256 whole neurons. For example, 64 units combine the incoming values into 64 outputs. More units add parameters and computation; they do not guarantee better accuracy.",
  ],
  filters: [
    "Filters",
    "The number of pattern detectors this convolutional layer learns: 1–64 whole filters. Eight filters produce eight feature maps. Early filters may respond to edges or strokes; later filters combine earlier features. Their jobs are learned, not assigned. More filters add computation and parameters.",
  ],
  kernel: [
    "Kernel size",
    "The width and height of the window each filter examines as it moves across its input. Choose 3 for a 3 × 3 window or 5 for 5 × 5. The first layer examines pixels; later layers examine feature maps. A larger window sees more at once and adds parameters, but is not always better.",
  ],
  activation: [
    "Activation",
    "The rule that transforms a unit’s output. ReLU keeps positive values and turns negative values into zero; tanh maps to −1…1; sigmoid maps to 0…1; linear leaves values unchanged. ReLU is the preset choice. Nonlinear activations help the network learn complex patterns; sigmoid and tanh can learn slowly when their outputs saturate.",
  ],
  rate: [
    "Drop rate",
    "The fraction of values randomly set to zero during training, from 0 to 0.8. For example, 0.2 drops about 20%. This can reduce overfitting, but too much can prevent learning. Dropout is off during prediction.",
  ],
  samples: [
    "Training images",
    "How many different MNIST training images each epoch uses: 1,000, 5,000, or 10,000. More images usually provide more variety and take longer. All runs use fixed, nested subsets and a separate fixed 1,000-image validation set.",
  ],
  epochs: [
    "Epochs",
    "One epoch is one complete pass through the selected training images. With 5,000 images and 3 epochs, each image is used three times. Choose a whole number from 1–30. More passes may improve learning or cause overfitting. The 30-epoch cap is an app limit, not a neural-network limit.",
  ],
  batch: [
    "Batch size",
    "How many images are processed before one weight update: 32, 64, or 128. With 1,000 images and a batch of 64 there are 16 updates per epoch; the last batch has 40 images. Larger batches use more memory and make fewer updates per epoch; speed and accuracy can vary.",
  ],
  lr: [
    "Learning rate",
    "Controls how large the weight adjustments are. Choose 0.01, 0.001, or 0.0001; 0.001 is the default. Too high can overshoot or become unstable; too low can learn slowly. Adam also adapts the adjustment for each weight.",
  ],
  "add-type": [
    "Layer type",
    "Dense combines all incoming values. Convolution learns local patterns. Max pooling shrinks feature maps. Dropout randomly suppresses values during training. Choose a type, then Add layer. Convolution and pooling must come before dense layers. At most 10 editable layers and 2 million parameters are allowed.",
  ],
  "save-name": [
    "Save name",
    "An optional name of up to 60 characters for a browser save, such as “Dense 64, first experiment”. Blank names get a timestamp. Names do not change the network, and duplicate names are allowed.",
  ],
  saved: [
    "Browser saves",
    "Select a previously saved experiment, then Load selected. Config-only saves restore the design. Weight saves also restore prediction. Saves belong to this browser and site address; they do not sync across devices. Loading does not restore comparison history or resume training.",
  ],
  import: [
    "Import configuration",
    "Choose a Neural Lab version 1 JSON configuration, no larger than 50,000 bytes. Import replaces the design and discards active unsaved weights. It does not import trained weights or training settings.",
  ],
  export: [
    "Export configuration",
    "Download the current layer design as JSON. The file contains architecture only, not trained weights, training settings, or run history. Import this file to recreate the design.",
  ],
  "save-config": [
    "Save configuration",
    "Save the current layer design under the chosen name in this browser. This does not save weights, training settings, or comparison history. Use Save weights + config to preserve a trained model.",
  ],
  "save-model": [
    "Save weights and configuration",
    "Save the active trained model and its design in this browser. Train or load a model first. Loading restores prediction, but Train new run always starts fresh. Clearing site data removes these saves; exported configuration JSON is not a weight backup.",
  ],
  load: [
    "Load selected",
    "Replace the current model with the selected browser save. Save your active weights first if you want to keep them. Weight saves restore prediction; configuration-only saves require training.",
  ],
  delete: [
    "Delete selected",
    "Remove the selected browser save after confirmation. This removes its stored configuration and weights, if present. It does not delete the currently active model.",
  ],
  dense: [
    "Dense preset",
    "Replace the design with 64 ReLU neurons and the automatic 10-class output. Pixels are flattened into 784 values first. This discards active unsaved weights, but keeps browser saves and comparison history.",
  ],
  cnn: [
    "Convolution preset",
    "Replace the design with 8 filters → pooling → 16 filters → pooling → 32 dense neurons → the automatic output. Both convolutions use 3 × 3 kernels and ReLU. This discards active unsaved weights.",
  ],
  pool: [
    "Max pooling",
    "For each feature map, keep the largest value in each 2 × 2 window and move by 2 positions. This roughly halves width and height, reducing computation. It has no learned weights. Window and stride are fixed at 2 in this app.",
  ],
  flatten: [
    "Automatic flatten",
    "Reshape the spatial feature maps into one list before the first dense layer. For example, 28 × 28 × 1 becomes 784 values. It changes the shape, not the values, and has no learned weights. Added automatically; there is no setting to change.",
  ],
  output: [
    "Automatic output",
    "The final dense layer has 10 units, one for each digit 0–9. Softmax converts their outputs into scores summing to 100%. The largest score is the prediction, not a guarantee of correctness. These 10 units and softmax are fixed.",
  ],
};
let serial = 0;
export function attachHelp(root = document) {
  const targets = [
    ...root.querySelectorAll("input, select, button[id], [data-help]"),
  ];
  for (const control of targets) {
    const key = control.dataset.help || control.dataset.key || control.id;
    if (!topics[key] || control.dataset.helpAttached) continue;
    control.dataset.helpAttached = "true";
    const [title, text] = topics[key];
    const help = document.createElement("span");
    help.className = "help";
    const id = `help-${++serial}`;
    help.innerHTML = `<button type="button" class="help-toggle" aria-label="About ${title}" aria-expanded="false" aria-controls="${id}" aria-describedby="${id}">?</button><span class="help-content" role="tooltip" id="${id}" hidden><strong>${title}</strong><span></span></span>`;
    help.querySelector(".help-content span").textContent = text;
    const button = help.querySelector("button"),
      panel = help.querySelector("[role=tooltip]");
    let pinned = false,
      hideTimer;
    const show = (value) => {
      panel.hidden = !value;
      button.setAttribute("aria-expanded", String(value));
      if (value) {
        const rect = button.getBoundingClientRect();
        panel.style.left = `${Math.max(8, Math.min(rect.left, document.documentElement.clientWidth - panel.offsetWidth - 8))}px`;
        panel.style.top = `${Math.max(8, Math.min(rect.bottom + 6, document.documentElement.clientHeight - panel.offsetHeight - 8))}px`;
      }
    };
    help.addEventListener("pointerenter", (e) => {
      clearTimeout(hideTimer);
      if (e.pointerType !== "touch") show(true);
    });
    help.addEventListener("pointerleave", () => {
      hideTimer = setTimeout(() => {
        if (!pinned && !help.contains(document.activeElement)) show(false);
      }, 150);
    });
    button.addEventListener("focus", () => show(true));
    help.addEventListener("focusout", (e) => {
      if (!help.contains(e.relatedTarget)) {
        pinned = false;
        show(false);
      }
    });
    button.addEventListener("click", (e) => {
      e.preventDefault();
      pinned = !pinned;
      show(pinned);
    });
    help.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        pinned = false;
        show(false);
        e.stopPropagation();
      }
    });
    // Keep the help button outside the associated label to avoid activating inputs.
    const label = control.closest("label");
    if (label && !label.classList.contains("file-button")) {
      control.id ||= `field-${++serial}`;
      const field = document.createElement("div");
      field.className = "field";
      const heading = document.createElement("div");
      heading.className = "field-heading";
      const caption = document.createElement("label");
      caption.htmlFor = control.id;
      caption.textContent = [...label.childNodes]
        .filter((n) => n.nodeType === 3)
        .map((n) => n.textContent)
        .join("");
      heading.append(caption, help);
      label.replaceWith(field);
      field.append(heading, control);
      if (control.type === "number") {
        const range = document.createElement("small");
        range.className = "range-hint";
        range.textContent = `${control.min}–${control.max}${key === "rate" ? "" : " · whole numbers"}`;
        field.append(range);
      }
    } else {
      const anchor = label || control;
      const wrapper = document.createElement("span");
      wrapper.className = "help-control";
      anchor.replaceWith(wrapper);
      wrapper.append(anchor, help);
    }
  }
}
