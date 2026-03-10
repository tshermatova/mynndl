// app.js
// Browser-only MNIST trainer using TensorFlow.js and tfjs-vis.
// - Loads train/test CSVs from local files (no network).
// - Builds and trains a CNN fully client-side.
// - Evaluates with confusion matrix and per-class accuracy in tfjs-vis visor.
// - Allows saving/loading model via files only (downloads and browserFiles).
(() => {
  'use strict';

  // DOM elements
  const els = {
    trainCsv: document.getElementById('trainCsv'),
    testCsv: document.getElementById('testCsv'),
    btnLoadData: document.getElementById('btnLoadData'),
    btnToggleVisor: document.getElementById('btnToggleVisor'),
    btnTrain: document.getElementById('btnTrain'),
    btnEvaluate: document.getElementById('btnEvaluate'),
    btnTestFive: document.getElementById('btnTestFive'),
    btnReset: document.getElementById('btnReset'),
    btnSave: document.getElementById('btnSave'),
    btnLoadModel: document.getElementById('btnLoadModel'),
    modelJson: document.getElementById('modelJson'),
    modelWeights: document.getElementById('modelWeights'),
    dataStatus: document.getElementById('dataStatus'),
    trainingLogs: document.getElementById('trainingLogs'),
    overallAcc: document.getElementById('overallAcc'),
    previewRow: document.getElementById('previewRow'),
    modelInfo: document.getElementById('modelInfo')
  };

  // Global state
  let model = null;
  let rawTrain = null;   // { xs: tf.Tensor4D, ys: tf.Tensor2D }
  let train = null;      // { xs: tf.Tensor4D, ys: tf.Tensor2D }
  let val = null;        // { xs: tf.Tensor4D, ys: tf.Tensor2D }
  let test = null;       // { xs: tf.Tensor4D, ys: tf.Tensor2D }
  let isTraining = false;

  const HYPER = {
    epochs: 8,
    batchSize: 128,
    valRatio: 0.1
  };

  // Utility: log to training logs panel
  function log(msg) {
    const ts = new Date().toLocaleTimeString();
    els.trainingLogs.textContent += `[${ts}] ${msg}\n`;
    els.trainingLogs.scrollTop = els.trainingLogs.scrollHeight;
  }

  // Utility: set button states to keep UI responsive and safe
  function setUIState({ loading = false, hasData = false, hasModel = false } = {}) {
    els.btnLoadData.disabled = loading;
    els.btnTrain.disabled = loading || !hasData;
    els.btnEvaluate.disabled = loading || !hasData || !hasModel;
    els.btnTestFive.disabled = loading || !hasData || !hasModel;
    els.btnSave.disabled = loading || !hasModel;
    els.btnReset.disabled = loading ? true : false;
    els.btnLoadModel.disabled = loading ? true : false;
  }

  // Cleanly dispose tensors/models
  function disposeTensorsGroup(obj) {
    if (!obj) return;
    for (const k of Object.keys(obj)) {
      if (obj[k] && typeof obj[k].dispose === 'function') {
        try { obj[k].dispose(); } catch (_) {}
      }
    }
  }
  function disposeAll() {
    disposeTensorsGroup(rawTrain);
    disposeTensorsGroup(train);
    disposeTensorsGroup(val);
    disposeTensorsGroup(test);
    rawTrain = train = val = test = null;
    if (model) { try { model.dispose(); } catch (_) {} model = null; }
    tf.engine().reset();
  }

  // Data status text
  function updateDataStatus() {
    const parts = [];
    if (rawTrain) parts.push(`Train: ${rawTrain.xs.shape[0]} samples`);
    if (val) parts.push(`Val: ${val.xs.shape[0]} samples`);
    if (test) parts.push(`Test: ${test.xs.shape[0]} samples`);
    els.dataStatus.textContent = parts.length ? parts.join(' • ') : 'No data loaded.';
  }

  // Build CNN model
  function buildModel() {
    const m = tf.sequential();
    m.add(tf.layers.conv2d({
      inputShape: [28, 28, 1],
      filters: 32, kernelSize: 3, activation: 'relu', padding: 'same'
    }));
    m.add(tf.layers.conv2d({
      filters: 64, kernelSize: 3, activation: 'relu', padding: 'same'
    }));
    m.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    m.add(tf.layers.dropout({ rate: 0.25 }));
    m.add(tf.layers.flatten());
    m.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    m.add(tf.layers.dropout({ rate: 0.5 }));
    m.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    m.compile({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    return m;
  }

  // Render model info (layers/params) into the Model Info section; also show summary in visor.
  function renderModelInfo(m) {
    if (!m) { els.modelInfo.textContent = 'No model.'; return; }
    // Total params
    const totalParams = m.weights.reduce((acc, w) => acc + w.shape.reduce((p, d) => p * d, 1), 0);
    const lines = [];
    lines.push(`Layers: ${m.layers.length}`);
    lines.push(`Total Params: ${totalParams.toLocaleString()}`);
    lines.push('');
    lines.push('Layer (type) → Output shape / Params');
    m.layers.forEach(layer => {
      const outShape = Array.isArray(layer.outputShape) ? JSON.stringify(layer.outputShape) : (layer.outputShape + '');
      const p = layer.countParams ? layer.countParams() : 0;
      lines.push(`• ${layer.name} (${layer.getClassName()}) → ${outShape} / ${p.toLocaleString()}`);
    });
    els.modelInfo.textContent = lines.join('\n');

    // Also show TFJS-VIS model summary
    try {
      tfvis.show.modelSummary({ name: 'Model Summary', tab: 'Model' }, m);
    } catch (err) {
      // non-fatal
    }
  }

  // Load Data: read both CSV files, build tensors, split into train/val, and show counts.
  async function onLoadData() {
    if (!window.MNISTData) {
    alert('MNISTData helper is not available. Make sure data-loader.js is loaded before app.js.');
    console.error('data-loader.js did not initialize window.MNISTData');
    return;
  }
    try {
      setUIState({ loading: true, hasData: !!rawTrain, hasModel: !!model });
      els.trainingLogs.textContent = '';
      log('Loading CSV files in browser...');

      const trainFile = els.trainCsv.files?.[0];
      const testFile = els.testCsv.files?.[0];
      if (!trainFile || !testFile) {
        throw new Error('Please select both Train and Test CSV files.');
      }

      // Dispose any previous data
      disposeTensorsGroup(rawTrain); disposeTensorsGroup(train); disposeTensorsGroup(val); disposeTensorsGroup(test);
      rawTrain = train = val = test = null;

      // Load tensors
      const [trainLoaded, testLoaded] = await Promise.all([
        window.MNISTData.loadTrainFromFiles(trainFile),
        window.MNISTData.loadTestFromFiles(testFile)
      ]);
      rawTrain = trainLoaded;
      test = testLoaded;

      // Split train/validation
      const { trainXs, trainYs, valXs, valYs } = window.MNISTData.splitTrainVal(rawTrain.xs, rawTrain.ys, HYPER.valRatio);
      train = { xs: trainXs, ys: trainYs };
      val = { xs: valXs, ys: valYs };

      updateDataStatus();
      log(`Loaded Train: ${rawTrain.xs.shape[0]} | Val: ${val.xs.shape[0]} | Test: ${test.xs.shape[0]}`);
      setUIState({ loading: false, hasData: true, hasModel: !!model });
    } catch (err) {
      console.error(err);
      log(`Error: ${err.message || err}`);
      setUIState({ loading: false, hasData: !!rawTrain, hasModel: !!model });
      alert(err.message || String(err));
    }
  }

  // Train with tfjs-vis live charts.
  async function onTrain() {
    if (!train || !val) { alert('Load data first.'); return; }
    if (isTraining) return;
    isTraining = true;
    setUIState({ loading: true, hasData: true, hasModel: !!model });
    try {
      // Build/replace model if not present
      if (model) { try { model.dispose(); } catch(_) {} }
      model = buildModel();
      renderModelInfo(model);

      const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
      const surface = { name: 'Training Metrics', tab: 'Training' };
      const fitCallbacks = tfvis.show.fitCallbacks(surface, metrics);

      let bestValAcc = 0, bestEpoch = -1;
      const t0 = performance.now();

      log(`Training started: epochs=${HYPER.epochs}, batchSize=${HYPER.batchSize}`);
      await model.fit(train.xs, train.ys, {
        validationData: [val.xs, val.ys],
        epochs: HYPER.epochs,
        batchSize: HYPER.batchSize,
        shuffle: true,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            await fitCallbacks.onEpochEnd(epoch, logs); // render charts
            const va = logs?.val_acc ?? logs?.val_accuracy ?? 0;
            if (va > bestValAcc) { bestValAcc = va; bestEpoch = epoch + 1; }
            log(`Epoch ${epoch + 1}/${HYPER.epochs}: loss=${(logs.loss||0).toFixed(4)}, acc=${(logs.acc||logs.accuracy||0).toFixed(4)}, val_loss=${(logs.val_loss||0).toFixed(4)}, val_acc=${(va||0).toFixed(4)}`);
            await tf.nextFrame(); // keep UI responsive
          },
          onTrainEnd: async () => {
            await fitCallbacks.onTrainEnd();
          }
        }
      });

      const ms = performance.now() - t0;
      log(`Training finished in ${(ms/1000).toFixed(2)}s. Best val_acc=${bestValAcc.toFixed(4)} @ epoch ${bestEpoch}`);
      setUIState({ loading: false, hasData: true, hasModel: true });
    } catch (err) {
      console.error(err);
      log(`Training Error: ${err.message || err}`);
      alert(err.message || String(err));
      setUIState({ loading: false, hasData: true, hasModel: !!model });
    } finally {
      isTraining = false;
    }
  }

  // Evaluate on test set: overall accuracy, confusion matrix, per-class accuracy.
  async function onEvaluate() {
    if (!model || !test) { alert('Train or load a model and load data first.'); return; }
    setUIState({ loading: true, hasData: true, hasModel: true });
    try {
      // Predict on test set in batches to avoid memory spikes
      const N = test.xs.shape[0];
      const bs = 256;
      const yTrue = [];
      const yPred = [];

      for (let i = 0; i < N; i += bs) {
        const size = Math.min(bs, N - i);
        const xsBatch = test.xs.slice([i, 0, 0, 0], [size, 28, 28, 1]);
        const ysBatch = test.ys.slice([i, 0], [size, 10]);
        const preds = tf.tidy(() => model.predict(xsBatch));

        const predLabels = preds.argMax(-1);
        const trueLabels = ysBatch.argMax(-1);
        const predArr = Array.from(await predLabels.data());
        const trueArr = Array.from(await trueLabels.data());
        yPred.push(...predArr);
        yTrue.push(...trueArr);

        predLabels.dispose();
        trueLabels.dispose();
        preds.dispose();
        xsBatch.dispose();
        ysBatch.dispose();
        await tf.nextFrame();
      }

      // Overall accuracy
      let correct = 0;
      for (let i = 0; i < N; i++) if (yTrue[i] === yPred[i]) correct++;
      const acc = correct / N;
      els.overallAcc.textContent = `${(acc * 100).toFixed(2)}% (${correct}/${N})`;

      // Confusion matrix 10x10
      const numClasses = 10;
      const cm = Array.from({ length: numClasses }, () => Array(numClasses).fill(0));
      for (let i = 0; i < N; i++) cm[yTrue[i]][yPred[i]]++;

      // Per-class accuracy
      const perClassAcc = [];
      for (let c = 0; c < numClasses; c++) {
        const row = cm[c];
        const total = row.reduce((a, b) => a + b, 0);
        const diag = row[c];
        perClassAcc.push({ index: c, label: String(c), acc: total ? diag / total : 0 });
      }

      // Render in visor
      const cmSurface = { name: 'Confusion Matrix', tab: 'Evaluation' };
      await tfvis.render.heatmap(cmSurface, { values: cm, xTickLabels: [...Array(10).keys()].map(String), yTickLabels: [...Array(10).keys()].map(String) }, { colorMap: 'blues', width: 420, height: 420 });

      const barSurface = { name: 'Per-class Accuracy', tab: 'Evaluation' };
      await tfvis.render.barchart(barSurface, perClassAcc.map(d => ({ x: d.label, y: d.acc })), {
        xLabel: 'Class', yLabel: 'Accuracy', width: 420, height: 300, yAxisDomain: [0, 1]
      });

      log(`Evaluation done. Overall accuracy: ${(acc * 100).toFixed(2)}%`);
    } catch (err) {
      console.error(err);
      log(`Evaluate Error: ${err.message || err}`);
      alert(err.message || String(err));
    } finally {
      setUIState({ loading: false, hasData: true, hasModel: true });
    }
  }

  // Preview 5 random test images with predicted labels (green if correct, red if wrong).
  async function onTestFive() {
    if (!model || !test) { alert('Train or load a model and load data first.'); return; }
    try {
      els.previewRow.innerHTML = '';
      const { xs: batchXs, ys: batchYs } = window.MNISTData.getRandomTestBatch(test.xs, test.ys, 5);
      const preds = tf.tidy(() => model.predict(batchXs));
      const predIdx = preds.argMax(-1);
      const trueIdx = batchYs.argMax(-1);

      const predArr = Array.from(await predIdx.data());
      const trueArr = Array.from(await trueIdx.data());

      for (let i = 0; i < predArr.length; i++) {
        const item = document.createElement('div');
        item.className = 'preview-item';
        const canvas = document.createElement('canvas');
        const label = document.createElement('div');
        label.className = 'pred';
        const ok = predArr[i] === trueArr[i];
        label.classList.add(ok ? 'ok' : 'bad');
        label.textContent = `Pred: ${predArr[i]} (True: ${trueArr[i]})`;

        // Draw image to canvas
        const img = batchXs.slice([i, 0, 0, 0], [1, 28, 28, 1]);
        await window.MNISTData.draw28x28ToCanvas(img, canvas, 4);
        img.dispose();

        item.appendChild(canvas);
        item.appendChild(label);
        els.previewRow.appendChild(item);
      }

      predIdx.dispose();
      trueIdx.dispose();
      preds.dispose();
      batchXs.dispose();
      batchYs.dispose();
      await tf.nextFrame();
    } catch (err) {
      console.error(err);
      log(`Preview Error: ${err.message || err}`);
    }
  }

  // Save model via file download
  async function onSaveDownload() {
    if (!model) { alert('No model to save.'); return; }
    try {
      await model.save('downloads://mnist-cnn');
      log('Model saved: downloaded model.json and weights.bin');
    } catch (err) {
      console.error(err);
      log(`Save Error: ${err.message || err}`);
      alert(err.message || String(err));
    }
  }

  // Load model from selected JSON and BIN files (no IndexedDB/LocalStorage)
  async function onLoadFromFiles() {
    const jsonFile = els.modelJson.files?.[0];
    const weightsFile = els.modelWeights.files?.[0];
    if (!jsonFile || !weightsFile) {
      alert('Please choose both model.json and weights.bin files.');
      return;
    }
    setUIState({ loading: true, hasData: !!rawTrain, hasModel: !!model });
    try {
      const loaded = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
      // Replace current model
      if (model) { try { model.dispose(); } catch (_) {} }
      model = loaded;
      // Re-compile (compile state isn't serialized)
      model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });
      renderModelInfo(model);
      log('Model loaded from files and compiled.');
      setUIState({ loading: false, hasData: !!rawTrain, hasModel: true });
    } catch (err) {
      console.error(err);
      log(`Load Model Error: ${err.message || err}`);
      alert(err.message || String(err));
      setUIState({ loading: false, hasData: !!rawTrain, hasModel: !!model });
    }
  }

  // Reset everything
  function onReset() {
    disposeAll();
    els.trainingLogs.textContent = '';
    els.dataStatus.textContent = 'No data loaded.';
    els.overallAcc.textContent = '—';
    els.previewRow.innerHTML = '';
    els.modelInfo.textContent = 'No model yet.';
    setUIState({ loading: false, hasData: false, hasModel: false });
    log('State reset.');
  }

  // Toggle tfjs-vis visor
  function onToggleVisor() {
    try { tfvis.visor().toggle(); } catch (_) { /* ignore */ }
  }

  // Wire up events
  function init() {
    els.btnLoadData.addEventListener('click', () => onLoadData());
    els.btnTrain.addEventListener('click', () => onTrain());
    els.btnEvaluate.addEventListener('click', () => onEvaluate());
    els.btnTestFive.addEventListener('click', () => onTestFive());
    els.btnSave.addEventListener('click', () => onSaveDownload());
    els.btnLoadModel.addEventListener('click', () => onLoadFromFiles());
    els.btnReset.addEventListener('click', () => onReset());
    els.btnToggleVisor.addEventListener('click', () => onToggleVisor());

    setUIState({ loading: false, hasData: false, hasModel: false });
    log('Ready. Upload CSVs and click "Load Data".');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
