// data-loader.js
// Browser-only CSV loader for MNIST (no external libraries).
// - CSV lines: label (0-9), then 784 pixels (0-255), no header.
// - Normalizes pixels to [0,1], reshapes to [N,28,28,1], and one-hots labels to depth 10.
// - Provides convenience helpers for splitting, random sampling, and drawing to canvas.

/**
 * Parse a CSV File object into MNIST tensors.
 * @param {File} file CSV file
 * @returns {Promise<{xs: tf.Tensor4D, ys: tf.Tensor2D}>}
 */
async function parseMnistCsvFileToTensors(file) {
  if (!(file instanceof File)) throw new Error('Expected a File for CSV parsing.');
  // Read entire file as text. This is fine for MNIST sizes; if needed, this can be
  // replaced with a streaming parser using a FileReader and chunk splitting.
  // Using .text() simplifies and is supported widely.
  const raw = await file.text();

  // Split into lines and filter out empties.
  const lines = raw.split(/\r?\n/).filter(l => l.trim().length > 0);

  // Pre-allocate typed arrays for efficiency.
  const numRows = lines.length;
  const numPixels = 28 * 28;
  const xsBuf = new Float32Array(numRows * numPixels);
  const labelsBuf = new Int32Array(numRows);

  // Parse each line: first value is label, next 784 are pixel values (0-255).
  // Normalize to [0,1] by dividing by 255.
  let row = 0;
  for (const line of lines) {
    const parts = line.split(',');
    // Ignore malformed lines.
    if (parts.length < 1 + numPixels) continue;

    const label = parseInt(parts[0], 10);
    labelsBuf[row] = Number.isFinite(label) ? label : 0;

    const base = row * numPixels;
    for (let i = 0; i < numPixels; i++) {
      const v = parseFloat(parts[i + 1]);
      xsBuf[base + i] = (Number.isFinite(v) ? v : 0) / 255.0;
    }
    row++;
  }

  // If there were malformed trailing lines, slice to the actual count.
  const effectiveRows = row;
  const xs2d = tf.tensor2d(xsBuf.subarray(0, effectiveRows * numPixels), [effectiveRows, numPixels], 'float32');
  const xs = xs2d.reshape([effectiveRows, 28, 28, 1]);

  const labels = tf.tensor1d(labelsBuf.subarray(0, effectiveRows), 'int32');
  const ys = tf.oneHot(labels, 10).toFloat();

  // Dispose intermediates we no longer need.
  xs2d.dispose();
  labels.dispose();

  return { xs, ys };
}

/**
 * Load train tensors from a CSV File.
 * @param {File} file
 * @returns {Promise<{xs: tf.Tensor4D, ys: tf.Tensor2D}>}
 */
async function loadTrainFromFiles(file) {
  return parseMnistCsvFileToTensors(file);
}

/**
 * Load test tensors from a CSV File.
 * @param {File} file
 * @returns {Promise<{xs: tf.Tensor4D, ys: tf.Tensor2D}>}
 */
async function loadTestFromFiles(file) {
  return parseMnistCsvFileToTensors(file);
}

/**
 * Split tensors into train/val sets by ratio.
 * @param {tf.Tensor4D} xs
 * @param {tf.Tensor2D} ys
 * @param {number} valRatio (0,1)
 * @returns {{trainXs: tf.Tensor4D, trainYs: tf.Tensor2D, valXs: tf.Tensor4D, valYs: tf.Tensor2D}}
 */
function splitTrainVal(xs, ys, valRatio = 0.1) {
  if (xs.shape[0] !== ys.shape[0]) throw new Error('splitTrainVal: xs and ys must have same first dimension.');
  const N = xs.shape[0];
  const v = Math.max(1, Math.floor(N * valRatio));
  const t = N - v;

  const trainXs = xs.slice([0, 0, 0, 0], [t, 28, 28, 1]);
  const valXs = xs.slice([t, 0, 0, 0], [v, 28, 28, 1]);
  const trainYs = ys.slice([0, 0], [t, 10]);
  const valYs = ys.slice([t, 0], [v, 10]);

  return { trainXs, trainYs, valXs, valYs };
}

/**
 * Sample k random items from test set without replacement.
 * @param {tf.Tensor4D} xs
 * @param {tf.Tensor2D} ys
 * @param {number} k
 * @returns {{xs: tf.Tensor4D, ys: tf.Tensor2D, indices: number[]}}
 */
function getRandomTestBatch(xs, ys, k = 5) {
  const N = xs.shape[0];
  const kk = Math.max(1, Math.min(k, N));
  const all = Array.from({ length: N }, (_, i) => i);

  // Fisher-Yates shuffle first kk entries
  for (let i = 0; i < kk; i++) {
    const j = i + Math.floor(Math.random() * (N - i));
    const tmp = all[i]; all[i] = all[j]; all[j] = tmp;
  }
  const indices = all.slice(0, kk);

  const idxTensor = tf.tensor1d(indices, 'int32');
  const batchXs = xs.gather(idxTensor);
  const batchYs = ys.gather(idxTensor);
  idxTensor.dispose();

  return { xs: batchXs, ys: batchYs, indices };
}

/**
 * Draw a 28x28 grayscale image tensor to a canvas, scaled up for visibility.
 * - Accepts shapes: [28,28], [28,28,1], or [1,28,28,1]
 * @param {tf.Tensor} imageTensor normalized in [0,1]
 * @param {HTMLCanvasElement} canvas
 * @param {number} scale integer scale factor (default 4)
 * @returns {Promise<void>}
 */
async function draw28x28ToCanvas(imageTensor, canvas, scale = 4) {
  if (!(canvas instanceof HTMLCanvasElement)) throw new Error('draw28x28ToCanvas: expected a canvas element');
  await tf.nextFrame();

  // Normalize input rank and shape to [28,28]
  const img = tf.tidy(() => {
    let t = imageTensor;
    if (t.rank === 4) t = t.squeeze(); // [1,28,28,1] -> [28,28,1]
    if (t.rank === 3 && t.shape[2] === 1) t = t.squeeze([2]); // [28,28,1] -> [28,28]
    if (t.rank !== 2) throw new Error(`draw28x28ToCanvas: unexpected rank ${t.rank}`);
    // Clip to [0,1] just in case.
    return t.clipByValue(0, 1);
  });

  // Use an offscreen 28x28 canvas to draw pixels, then scale up on target canvas.
  const off = document.createElement('canvas');
  off.width = 28; off.height = 28;
  await tf.browser.toPixels(img, off);

  const ctx = canvas.getContext('2d');
  const W = 28 * scale, H = 28 * scale;
  canvas.width = W; canvas.height = H;
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(off, 0, 0, W, H);

  img.dispose();
}

// Expose to app.js
window.MNISTData = {
  loadTrainFromFiles,
  loadTestFromFiles,
  splitTrainVal,
  getRandomTestBatch,
  draw28x28ToCanvas
};
