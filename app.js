// app.js
// Denoising Autoencoder demo (browser-only):
// - Step 1: Add Gaussian noise to data (inputs only), keep clean targets.
// - Step 2: Train two CNN autoencoders: one with MaxPooling, one with AveragePooling.
// - Step 3: "Test 5 Random": show Original | Noisy | Denoised(Max) | Denoised(Avg).
// - Step 4: Save/reload selected model via files to reproduce results.

(() => {
  'use strict';

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
    modelSlot: document.getElementById('modelSlot'),
    dataStatus: document.getElementById('dataStatus'),
    trainingLogs: document.getElementById('trainingLogs'),
    psnrMax: document.getElementById('psnrMax'),
    psnrAvg: document.getElementById('psnrAvg'),
    previewRow: document.getElementById('previewRow'),
    modelInfo: document.getElementById('modelInfo'),
    noiseLevel: document.getElementById('noiseLevel'),
    noiseLevelNum: document.getElementById('noiseLevelNum')
  };

  let modelMax = null; // Autoencoder with MaxPooling
  let modelAvg = null; // Autoencoder with AveragePooling

  // Data tensors
  let rawTrain = null; // { xs, ys } (ys not used for AE)
  let train = null;    // { xs }
  let val = null;      // { xs }
  let test = null;     // { xs }

  let isTraining = false;

  const HYPER = {
    epochs: 6,
    batchSize: 128,
    valRatio: 0.1
  };

  function log(msg) {
    const ts = new Date().toLocaleTimeString();
    els.trainingLogs.textContent += `[${ts}] ${msg}\n`;
    els.trainingLogs.scrollTop = els.trainingLogs.scrollHeight;
  }

  function setUIState({ loading = false, hasData = false, hasAnyModel = false } = {}) {
    els.btnLoadData.disabled = loading;
    els.btnTrain.disabled = loading || !hasData;
    els.btnEvaluate.disabled = loading || !hasData || !hasAnyModel;
    els.btnTestFive.disabled = loading || !hasData || !hasAnyModel;
    els.btnSave.disabled = loading || !hasAnyModel;
    els.btnReset.disabled = loading ? true : false;
    els.btnLoadModel.disabled = loading ? true : false;
  }

  function disposeTensorsGroup(obj) {
    if (!obj) return;
    for (const k of Object.keys(obj)) {
      if (obj[k] && typeof obj[k].dispose === 'function') {
        try { obj[k].dispose(); } catch (_) {}
      }
    }
  }
  function disposeAll() {
    disposeTensorsGroup(rawTrain); rawTrain = null;
    disposeTensorsGroup(train); train = null;
    disposeTensorsGroup(val); val = null;
    disposeTensorsGroup(test); test = null;
    if (modelMax) { try { modelMax.dispose(); } catch (_) {} modelMax = null; }
    if (modelAvg) { try { modelAvg.dispose(); } catch (_) {} modelAvg = null; }
    tf.engine().reset();
  }

  function updateDataStatus() {
    const parts = [];
    if (train) parts.push(`Train: ${train.xs.shape[0]} samples`);
    if (val) parts.push(`Val: ${val.xs.shape[0]} samples`);
    if (test) parts.push(`Test: ${test.xs.shape[0]} samples`);
    els.dataStatus.textContent = parts.length ? parts.join(' • ') : 'No data loaded.';
  }

  // Noise utilities
  function getNoiseSigma() {
    const v = parseFloat(els.noiseLevel.value);
    return Number.isFinite(v) ? Math.max(0, Math.min(0.6, v)) : 0.3;
  }
  function syncNoiseInputs() {
    els.noiseLevelNum.value = getNoiseSigma().toFixed(2);
    els.noiseLevel.value = els.noiseLevelNum.value;
  }
  // Add zero-mean Gaussian noise, clip to [0,1]
  function addGaussianNoise(batch, sigma) {
    return tf.tidy(() => {
      if (sigma <= 0) return batch.clone();
      const noise = tf.randomNormal(batch.shape, 0, sigma, 'float32');
      return batch.add(noise).clipByValue(0, 1);
    });
  }

  // Build a CNN autoencoder; poolType: 'max' | 'avg'
  function buildAutoencoder(poolType = 'max') {
    const m = tf.sequential();
    const poolLayer = poolType === 'avg'
      ? tf.layers.averagePooling2d({ poolSize: [2, 2] })
      : tf.layers.maxPooling2d({ poolSize: [2, 2] });

    // Encoder
    m.add(tf.layers.conv2d({ inputShape: [28, 28, 1], filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }));
    m.add(poolLayer);
    m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
    m.add(poolLayer.clone ? poolLayer.clone() : (poolType === 'avg' ? tf.layers.averagePooling2d({ poolSize: [2, 2] }) : tf.layers.maxPooling2d({ poolSize: [2, 2] })));
    // Bottleneck
    m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
    // Decoder
    m.add(tf.layers.upSampling2d({ size: [2, 2] }));
    m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
    m.add(tf.layers.upSampling2d({ size: [2, 2] }));
    m.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }));
    m.add(tf.layers.conv2d({ filters: 1, kernelSize: 3, activation: 'sigmoid', padding: 'same' }));

    m.compile({
      optimizer: 'adam',
      loss: 'meanSquaredError',
      metrics: ['mae']
    });
    return m;
  }

  // Show info for both models
  function renderModelInfo() {
    const lines = [];
    const addInfo = (label, m) => {
      if (!m) { lines.push(`• ${label}: (not initialized)`); return; }
      const totalParams = m.weights.reduce((acc, w) => acc + w.shape.reduce((p, d) => p * d, 1), 0);
      lines.push(`• ${label}: Layers=${m.layers.length}, Params=${totalParams.toLocaleString()}`);
    };
    addInfo('MaxPool AE', modelMax);
    addInfo('AvgPool AE', modelAvg);
    els.modelInfo.textContent = lines.join('\n');
    try {
      if (modelMax) tfvis.show.modelSummary({ name: 'Model Summary (MaxPool AE)', tab: 'Model' }, modelMax);
      if (modelAvg) tfvis.show.modelSummary({ name: 'Model Summary (AvgPool AE)', tab: 'Model' }, modelAvg);
    } catch (_) {}
  }

  // Dataset generator that creates on-the-fly noisy inputs with clean targets.
  function makeNoisyDataset(xsClean, batchSize, sigma) {
    const N = xsClean.shape[0];
    const stepsPerEpoch = Math.ceil(N / batchSize);

    function* indexBatches() {
      // Shuffle indices each epoch
      const idx = Array.from({ length: N }, (_, i) => i);
      for (let i = N - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
      }
      for (let s = 0; s < stepsPerEpoch; s++) {
        const start = s * batchSize;
        const size = Math.min(batchSize, N - start);
        const batchIdx = idx.slice(start, start + size);
        yield batchIdx;
      }
    }

    const ds = tf.data.generator(function* () {
      for (const batchIdx of indexBatches()) {
        const idxTensor = tf.tensor1d(batchIdx, 'int32');
        const clean = xsClean.gather(idxTensor);  // [B,28,28,1]
        const noisy = addGaussianNoise(clean, sigma);
        idxTensor.dispose();
        yield { xs: noisy, ys: clean };
        // clean/noisy disposed by tfjs internally after each yield consumption
      }
    });

    return { dataset: ds, stepsPerEpoch };
  }

  // Validation dataset (deterministic, one pass)
  function makeValidationDataset(xsClean, batchSize, sigma) {
    const N = xsClean.shape[0];
    const steps = Math.ceil(N / batchSize);
    const ds = tf.data.generator(function* () {
      for (let i = 0; i < N; i += batchSize) {
        const size = Math.min(batchSize, N - i);
        const clean = xsClean.slice([i, 0, 0, 0], [size, 28, 28, 1]);
        const noisy = addGaussianNoise(clean, sigma);
        yield { xs: noisy, ys: clean };
      }
    });
    return { dataset: ds, steps };
  }

  // PSNR helper: with MAX_I = 1.0
  function batchPSNR(yTrue, yPred) {
    return tf.tidy(() => {
      const mse = tf.mean(tf.square(yPred.sub(yTrue)), [1, 2, 3]); // per-sample MSE
      const psnr = tf.mul(-10, tf.log(mse).div(Math.log(10))); // -10 * log10(mse)
      return psnr; // shape [B]
    });
  }

  async function onLoadData() {
    try {
      if (!window.MNISTData) {
        alert('MNISTData helper is not available. Make sure data-loader.js is loaded before app.js.');
        return;
      }
      setUIState({ loading: true, hasData: !!train, hasAnyModel: !!(modelMax || modelAvg) });
      els.trainingLogs.textContent = '';
      log('Loading CSV files in browser...');

      const trainFile = els.trainCsv.files?.[0];
      const testFile = els.testCsv.files?.[0];
      if (!trainFile || !testFile) {
        throw new Error('Please select both Train and Test CSV files.');
      }

      disposeTensorsGroup(rawTrain); disposeTensorsGroup(train); disposeTensorsGroup(val); disposeTensorsGroup(test);
      rawTrain = train = val = test = null;

      const [trainLoaded, testLoaded] = await Promise.all([
        window.MNISTData.loadTrainFromFiles(trainFile),
        window.MNISTData.loadTestFromFiles(testFile)
      ]);
      rawTrain = trainLoaded;
      test = { xs: testLoaded.xs }; // labels not needed for AE

      const { trainXs, trainYs, valXs, valYs } = window.MNISTData.splitTrainVal(rawTrain.xs, rawTrain.ys, HYPER.valRatio);
      // For AE we only need xs (clean targets); ys only used for split sizes
      train = { xs: trainXs };
      val = { xs: valXs };
      // Dispose label tensors from split
      trainYs.dispose();
      valYs.dispose();

      updateDataStatus();
      log(`Loaded Train: ${train.xs.shape[0]} | Val: ${val.xs.shape[0]} | Test: ${test.xs.shape[0]}`);
      setUIState({ loading: false, hasData: true, hasAnyModel: !!(modelMax || modelAvg) });
    } catch (err) {
      console.error(err);
      log(`Error: ${err.message || err}`);
      setUIState({ loading: false, hasData: !!train, hasAnyModel: !!(modelMax || modelAvg) });
      alert(err.message || String(err));
    }
  }

  async function onTrain() {
    if (!train || !val) { alert('Load data first.'); return; }
    if (isTraining) return;
    isTraining = true;
    setUIState({ loading: true, hasData: true, hasAnyModel: !!(modelMax || modelAvg) });
    try {
      // Rebuild models
      if (modelMax) { try { modelMax.dispose(); } catch(_) {} }
      if (modelAvg) { try { modelAvg.dispose(); } catch(_) {} }
      modelMax = buildAutoencoder('max');
      modelAvg = buildAutoencoder('avg');
      renderModelInfo();

      const sigma = getNoiseSigma();
      const batchSize = HYPER.batchSize;

      // Prepare datasets
      const trainDSMax = makeNoisyDataset(train.xs, batchSize, sigma);
      const valDSMax = makeValidationDataset(val.xs, batchSize, sigma);

      const trainDSAvg = makeNoisyDataset(train.xs, batchSize, sigma);
      const valDSAvg = makeValidationDataset(val.xs, batchSize, sigma);

      // Train MaxPool AE
      log(`Training MaxPool AE: epochs=${HYPER.epochs}, batchSize=${batchSize}, σ=${sigma}`);
      await modelMax.fitDataset(trainDSMax.dataset, {
        epochs: HYPER.epochs,
        batchesPerEpoch: trainDSMax.stepsPerEpoch,
        validationData: valDSMax.dataset,
        validationBatches: valDSMax.steps,
        callbacks: tfvis.show.fitCallbacks({ name: 'Training (MaxPool AE)', tab: 'Training (Max)' }, ['loss', 'val_loss', 'mae', 'val_mae'])
      });

      // Train AvgPool AE
      log(`Training AvgPool AE: epochs=${HYPER.epochs}, batchSize=${batchSize}, σ=${sigma}`);
      await modelAvg.fitDataset(trainDSAvg.dataset, {
        epochs: HYPER.epochs,
        batchesPerEpoch: trainDSAvg.stepsPerEpoch,
        validationData: valDSAvg.dataset,
        validationBatches: valDSAvg.steps,
        callbacks: tfvis.show.fitCallbacks({ name: 'Training (AvgPool AE)', tab: 'Training (Avg)' }, ['loss', 'val_loss', 'mae', 'val_mae'])
      });

      log('Training finished for both models.');
      renderModelInfo();
      setUIState({ loading: false, hasData: true, hasAnyModel: true });
    } catch (err) {
      console.error(err);
      log(`Training Error: ${err.message || err}`);
      alert(err.message || String(err));
      setUIState({ loading: false, hasData: true, hasAnyModel: !!(modelMax || modelAvg) });
    } finally {
      isTraining = false;
    }
  }

  async function onEvaluate() {
    if (!test) { alert('Load data first.'); return; }
    if (!modelMax && !modelAvg) { alert('Train or load at least one model.'); return; }
    setUIState({ loading: true, hasData: true, hasAnyModel: !!(modelMax || modelAvg) });
    try {
      const N = test.xs.shape[0];
      const bs = 256;
      const sigma = getNoiseSigma();

      async function evalModel(m) {
        if (!m) return null;
        let psnrSum = 0, count = 0;
        for (let i = 0; i < N; i += bs) {
          const size = Math.min(bs, N - i);
          const clean = test.xs.slice([i, 0, 0, 0], [size, 28, 28, 1]);
          const noisy = addGaussianNoise(clean, sigma);
          const pred = tf.tidy(() => m.predict(noisy));

          const ps = batchPSNR(clean, pred);
          const arr = await ps.data();
          for (let k = 0; k < arr.length; k++) { if (Number.isFinite(arr[k])) { psnrSum += arr[k]; count++; } }

          ps.dispose(); pred.dispose(); noisy.dispose(); clean.dispose();
          await tf.nextFrame();
        }
        return count ? psnrSum / count : null;
      }

      const [psnrM, psnrA] = await Promise.all([evalModel(modelMax), evalModel(modelAvg)]);
      els.psnrMax.textContent = psnrM ? `${psnrM.toFixed(2)} dB` : '—';
      els.psnrAvg.textContent = psnrA ? `${psnrA.toFixed(2)} dB` : '—';
      log(`Evaluation done. PSNR (MaxPool): ${psnrM ? psnrM.toFixed(2) : '—'} dB | PSNR (AvgPool): ${psnrA ? psnrA.toFixed(2) : '—'} dB`);
    } catch (err) {
      console.error(err);
      log(`Evaluate Error: ${err.message || err}`);
      alert(err.message || String(err));
    } finally {
      setUIState({ loading: false, hasData: true, hasAnyModel: !!(modelMax || modelAvg) });
    }
  }

  async function onTestFive() {
    if (!test) { alert('Load data first.'); return; }
    if (!modelMax && !modelAvg) { alert('Train or load at least one model.'); return; }
    try {
      els.previewRow.innerHTML = '';
      const { xs: batchClean } = window.MNISTData.getRandomTestBatch(test.xs, test.xs, 5); // ys not used
      const sigma = getNoiseSigma();
      const batchNoisy = addGaussianNoise(batchClean, sigma);

      let predsMax = null, predsAvg = null;
      if (modelMax) predsMax = tf.tidy(() => modelMax.predict(batchNoisy));
      if (modelAvg) predsAvg = tf.tidy(() => modelAvg.predict(batchNoisy));

      const num = batchClean.shape[0];
      for (let i = 0; i < num; i++) {
        const item = document.createElement('div');
        item.className = 'preview-item';

        const grid = document.createElement('div');
        grid.className = 'thumbs';

        // Helper to add a canvas+caption cell
        const addThumb = async (tensor, title) => {
          const wrap = document.createElement('div');
          wrap.className = 'thumb';
          const canvas = document.createElement('canvas');
          await window.MNISTData.draw28x28ToCanvas(tensor, canvas, 4);
          const cap = document.createElement('div');
          cap.className = 'caption';
          cap.textContent = title;
          wrap.appendChild(canvas);
          wrap.appendChild(cap);
          grid.appendChild(wrap);
        };

        const clean = batchClean.slice([i, 0, 0, 0], [1, 28, 28, 1]);
        const noisy = batchNoisy.slice([i, 0, 0, 0], [1, 28, 28, 1]);

        await addThumb(clean, 'Original');
        await addThumb(noisy, `Noisy (σ=${sigma.toFixed(2)})`);

        if (predsMax) {
          const den = predsMax.slice([i, 0, 0, 0], [1, 28, 28, 1]);
          await addThumb(den, 'Denoised (MaxPool)');
          den.dispose();
        } else {
          const ph = noisy.mul(0).add(0); await addThumb(ph, 'Denoised (MaxPool) — N/A'); ph.dispose();
        }

        if (predsAvg) {
          const den = predsAvg.slice([i, 0, 0, 0], [1, 28, 28, 1]);
          await addThumb(den, 'Denoised (AvgPool)');
          den.dispose();
        } else {
          const ph = noisy.mul(0).add(0); await addThumb(ph, 'Denoised (AvgPool) — N/A'); ph.dispose();
        }

        clean.dispose(); noisy.dispose();
        item.appendChild(grid);
        els.previewRow.appendChild(item);
      }

      if (predsMax) predsMax.dispose();
      if (predsAvg) predsAvg.dispose();
      batchNoisy.dispose();
      batchClean.dispose();
      await tf.nextFrame();
    } catch (err) {
      console.error(err);
      log(`Preview Error: ${err.message || err}`);
    }
  }

  async function onSaveDownload() {
    const slot = els.modelSlot.value; // 'max' | 'avg'
    const m = slot === 'max' ? modelMax : modelAvg;
    if (!m) { alert(`No model in slot "${slot.toUpperCase()}" to save.`); return; }
    try {
      await m.save(`downloads://mnist-ae-${slot}`);
      log(`Saved ${slot.toUpperCase()} model: downloaded model.json and weights.bin`);
    } catch (err) {
      console.error(err);
      log(`Save Error: ${err.message || err}`);
      alert(err.message || String(err));
    }
  }

  async function onLoadFromFiles() {
    const jsonFile = els.modelJson.files?.[0];
    const weightsFile = els.modelWeights.files?.[0];
    const slot = els.modelSlot.value; // 'max' | 'avg'
    if (!jsonFile || !weightsFile) {
      alert('Please choose both model.json and weights.bin files.');
      return;
    }
    setUIState({ loading: true, hasData: !!train, hasAnyModel: !!(modelMax || modelAvg) });
    try {
      const loaded = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
      // Re-compile (compile state isn't serialized)
      loaded.compile({ optimizer: 'adam', loss: 'meanSquaredError', metrics: ['mae'] });

      if (slot === 'max') {
        if (modelMax) { try { modelMax.dispose(); } catch (_) {} }
        modelMax = loaded;
      } else {
        if (modelAvg) { try { modelAvg.dispose(); } catch (_) {} }
        modelAvg = loaded;
      }

      renderModelInfo();
      log(`Loaded model into slot "${slot.toUpperCase()}" from files and compiled.`);
      setUIState({ loading: false, hasData: !!train, hasAnyModel: !!(modelMax || modelAvg) });
    } catch (err) {
      console.error(err);
      log(`Load Model Error: ${err.message || err}`);
      alert(err.message || String(err));
      setUIState({ loading: false, hasData: !!train, hasAnyModel: !!(modelMax || modelAvg) });
    }
  }

  function onReset() {
    disposeAll();
    els.trainingLogs.textContent = '';
    els.dataStatus.textContent = 'No data loaded.';
    els.psnrMax.textContent = '—';
    els.psnrAvg.textContent = '—';
    els.previewRow.innerHTML = '';
    els.modelInfo.textContent = 'No models yet.';
    setUIState({ loading: false, hasData: false, hasAnyModel: false });
    log('State reset.');
  }

  function onToggleVisor() {
    try { tfvis.visor().toggle(); } catch (_) {}
  }

  function init() {
    els.btnLoadData.addEventListener('click', onLoadData);
    els.btnTrain.addEventListener('click', onTrain);
    els.btnEvaluate.addEventListener('click', onEvaluate);
    els.btnTestFive.addEventListener('click', onTestFive);
    els.btnSave.addEventListener('click', onSaveDownload);
    els.btnLoadModel.addEventListener('click', onLoadFromFiles);
    els.btnReset.addEventListener('click', onReset);
    els.btnToggleVisor.addEventListener('click', onToggleVisor);

    // Sync noise controls
    els.noiseLevel.addEventListener('input', () => {
      els.noiseLevelNum.value = parseFloat(els.noiseLevel.value).toFixed(2);
    });
    els.noiseLevelNum.addEventListener('input', () => {
      const v = Math.max(0, Math.min(0.6, parseFloat(els.noiseLevelNum.value) || 0));
      els.noiseLevel.value = v.toFixed(2);
      els.noiseLevelNum.value = v.toFixed(2);
    });
    syncNoiseInputs();

    setUIState({ loading: false, hasData: false, hasAnyModel: false });
    log('Ready. Upload CSVs and click "Load Data".');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
