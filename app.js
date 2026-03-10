/* app.js – UI wiring, model definition (denoising + classifier), training & eval */

(async function() {
    // ----- DOM elements -----
    const trainFileInp = document.getElementById('train-file');
    const testFileInp = document.getElementById('test-file');
    const loadDataBtn = document.getElementById('load-data-btn');
    const trainBtn = document.getElementById('train-btn');
    const evaluateBtn = document.getElementById('evaluate-btn');
    const testFiveBtn = document.getElementById('test-five-btn');
    const saveModelBtn = document.getElementById('save-model-btn');
    const loadModelJsonFile = document.getElementById('model-json-file');
    const loadModelBtn = document.getElementById('load-model-btn');
    const resetBtn = document.getElementById('reset-btn');
    const toggleVisorBtn = document.getElementById('toggle-visor-btn');
    const dataStatusDiv = document.getElementById('data-status');
    const logDiv = document.getElementById('log-area');
    const metricsDiv = document.getElementById('metrics-display');
    const modelInfoDiv = document.getElementById('model-info');
    const previewContainer = document.getElementById('preview-container');

    // ----- state -----
    let trainXs, trainYs, testXs, testYs, valXs, valYs;
    let currentModel = null;
    let trainingEpochs = 5;      // default

    // helper log
    function log(message) {
        const p = document.createElement('div');
        p.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logDiv.appendChild(p);
        logDiv.scrollTop = logDiv.scrollHeight;
    }

    // clear log
    function clearLog() { logDiv.innerHTML = ''; }

    // update status
    function setDataStatus(trainCount, testCount) {
        dataStatusDiv.innerText = `train: ${trainCount} samples | test: ${testCount} samples`;
    }

    // ----- build denoising autoencoder + classifier (multi‑output) -----
    function buildDenoisingModel() {
        const input = tf.input({ shape: [28, 28, 1] });

        // Encoder (shared)
        const conv1 = tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(input);
        const conv2 = tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(conv1);
        const pool1 = tf.layers.maxPooling2d({ poolSize: 2 }).apply(conv2);         // [14,14,64]

        // ----- two decoder branches for denoising -----
        // branch A: maxpool upsampling (simple)
        const upsampleMax = tf.layers.upSampling2d({ size: [2, 2] }).apply(pool1);   // [28,28,64]
        const convMax1 = tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(upsampleMax);
        const reconMax = tf.layers.conv2d({ filters: 1, kernelSize: 3, activation: 'sigmoid', padding: 'same', name: 'recon_max' }).apply(convMax1);

        // branch B: average pooling + upsampling (simulate avgpool via average pooling?)
        // we use an average pooling layer then upsample.
        const avgPool = tf.layers.averagePooling2d({ poolSize: 2 }).apply(conv2);    // [14,14,64]
        const upsampleAvg = tf.layers.upSampling2d({ size: [2, 2] }).apply(avgPool); // [28,28,64]
        const convAvg1 = tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(upsampleAvg);
        const reconAvg = tf.layers.conv2d({ filters: 1, kernelSize: 3, activation: 'sigmoid', padding: 'same', name: 'recon_avg' }).apply(convAvg1);

        // ----- classifier head (from pooled features) -----
        const flat = tf.layers.flatten().apply(pool1);
        const drop1 = tf.layers.dropout({ rate: 0.25 }).apply(flat);
        const dense = tf.layers.dense({ units: 128, activation: 'relu' }).apply(drop1);
        const drop2 = tf.layers.dropout({ rate: 0.5 }).apply(dense);
        const classifier = tf.layers.dense({ units: 10, activation: 'softmax', name: 'classifier' }).apply(drop2);

        const model = tf.model({ inputs: input, outputs: [classifier, reconMax, reconAvg] });

        model.compile({
            optimizer: 'adam',
            loss: {
                classifier: 'categoricalCrossentropy',
                recon_max: 'meanSquaredError',
                recon_avg: 'meanSquaredError'
            },
            metrics: {
                classifier: ['accuracy'],
                recon_max: ['mse'],
                recon_avg: ['mse']
            },
            loss_weights: { classifier: 1.0, recon_max: 0.5, recon_avg: 0.5 }  // focus on classification
        });
        return model;
    }

    // ----- reset everything -----
    function resetAll() {
        if (trainXs) { trainXs.dispose(); trainYs.dispose(); }
        if (testXs) { testXs.dispose(); testYs.dispose(); }
        if (valXs) { valXs.dispose(); valYs.dispose(); }
        if (currentModel) { currentModel.dispose(); }
        trainXs = trainYs = testXs = testYs = valXs = valYs = null;
        currentModel = null;
        setDataStatus(0, 0);
        modelInfoDiv.innerText = 'Model reset. Build on train.';
        previewContainer.innerHTML = '<span style="color:#6b7280;">reset — load data & train</span>';
        metricsDiv.innerText = 'overall accuracy: —';
        log('🔥 reset complete');
    }

    // ----- load data -----
    loadDataBtn.addEventListener('click', async () => {
        try {
            if (!trainFileInp.files[0] || !testFileInp.files[0]) {
                alert('Please select both train and test CSV files');
                return;
            }
            log('📂 loading train file...');
            const train = await dataLoader.loadTrainFromFiles(trainFileInp.files[0]);
            trainXs = train.xs; trainYs = train.ys;

            log('📂 loading test file...');
            const test = await dataLoader.loadTestFromFiles(testFileInp.files[0]);
            testXs = test.xs; testYs = test.ys;

            // split train into train/val (10% val)
            const split = dataLoader.splitTrainVal(trainXs, trainYs, 0.1);
            trainXs = split.trainXs; trainYs = split.trainYs;
            valXs = split.valXs; valYs = split.valYs;

            setDataStatus(trainXs.shape[0], testXs.shape[0]);
            log(`✅ train: ${trainXs.shape[0]}, val: ${valXs.shape[0]}, test: ${testXs.shape[0]}`);
        } catch (e) {
            log(`❌ load error: ${e.message}`);
            console.error(e);
        }
    });

    // ----- train -----
    trainBtn.addEventListener('click', async () => {
        if (!trainXs) { alert('load data first'); return; }
        try {
            log('🏗️ building denoising model...');
            if (currentModel) currentModel.dispose();
            currentModel = buildDenoisingModel();
            currentModel.summary();

            // Prepare training with tfvis callbacks
            const container = { name: 'Training Loss & Acc', tab: 'Training' };
            const callbacks = tfvis.show.fitCallbacks(container, ['classifier_loss', 'classifier_acc', 'recon_max_loss', 'recon_avg_loss'], {
                height: 300, callbacks: ['onEpochEnd']
            });

            log('🚀 training started (5 epochs, batch 128)...');
            const start = performance.now();
            const history = await currentModel.fit(trainXs, {
                classifier: trainYs,
                recon_max: trainXs,   // target = original clean image
                recon_avg: trainXs
            }, {
                batchSize: 128,
                epochs: 5,
                validationData: [valXs, { classifier: valYs, recon_max: valXs, recon_avg: valXs }],
                shuffle: true,
                callbacks: callbacks
            });
            const duration = ((performance.now() - start) / 1000).toFixed(2);
            log(`✅ training finished in ${duration}s`);

            const lastAcc = history.history.classifier_acc.slice(-1)[0];
            metricsDiv.innerText = `overall accuracy: ${(lastAcc*100).toFixed(2)}% (last batch)`;

            modelInfoDiv.innerText = `CNN autoencoder trained. layers: conv, pool, two decoders.`;
        } catch (e) {
            log(`❌ train error: ${e.message}`);
        }
    });

    // ----- evaluate: overall accuracy + confusion matrix + per‑class bar chart -----
    evaluateBtn.addEventListener('click', async () => {
        if (!currentModel || !testXs) { alert('train model & load test data first'); return; }
        try {
            log('📊 evaluating on test set (noisy not added)');
            const evalOutput = await currentModel.evaluate(testXs, {
                classifier: testYs,
                recon_max: testXs,
                recon_avg: testXs
            }, { batchSize: 128 });
            // evalOutput is array of losses & metrics [classifier_loss, recon_max_loss, recon_avg_loss, classifier_acc, recon_max_mse, recon_avg_mse]
            const acc = evalOutput[3].dataSync()[0]; // classifier acc
            metricsDiv.innerText = `✅ test accuracy: ${(acc*100).toFixed(2)}%`;

            // Confusion matrix and per-class using tfvis
            const preds = currentModel.predict(testXs)[0].argMax(-1);
            const labels = testYs.argMax(-1);
            await tfvis.metrics.confusionMatrix({
                values: await preds.array(),
                labels: await labels.array(),
                classNames: ['0','1','2','3','4','5','6','7','8','9']
            });
            // per-class accuracy
            const perClassAcc = tf.tidy(() => {
                const eq = preds.equal(labels).cast('float32');
                const totalPerClass = tf.matMul(tf.oneHot(labels, 10).cast('float32').transpose(), tf.ones([labels.shape[0], 1]));
                const correctPerClass = tf.matMul(tf.oneHot(labels, 10).cast('float32').transpose(), eq.reshape([-1,1]));
                return correctPerClass.div(totalPerClass).arraySync();
            });
            const perClass = await perClassAcc;
            const data = perClass.map((v,i) => ({ index: i, class: i.toString(), accuracy: v[0] }));
            tfvis.render.barchart({ name: 'per‑class accuracy', tab: 'Evaluation' }, data, { xLabel: 'class' });

            log('evaluation charts added to visor');
        } catch (e) {
            log(`❌ eval error: ${e.message}`);
        }
    });

    // ----- test 5 random: add noise, denoise with both branches, show original/noisy/recon -----
    testFiveBtn.addEventListener('click', async () => {
        if (!currentModel || !testXs) { alert('load test data and train model first'); return; }
        try {
            // get 5 random test images (clean)
            const batch = dataLoader.getRandomTestBatch(testXs, testYs, 5);
            const cleanImgs = batch.xs;      // [5,28,28,1]
            const trueLabels = batch.ys.argMax(-1).arraySync(); // [5]

            // add noise (level 0.3)
            const noisyImgs = dataLoader.addNoiseToTensors(cleanImgs, 0.3);

            // run through model: outputs = [class_pred, recon_max, recon_avg]
            const outputs = currentModel.predict(noisyImgs);
            const classPred = outputs[0].argMax(-1).arraySync(); // [5]
            const reconMax = outputs[1];      // [5,28,28,1]
            const reconAvg = outputs[2];

            // clear preview
            previewContainer.innerHTML = '';

            // for each of 5 samples, create canvas triplet (noisy, max recon, avg recon) and label
            for (let i = 0; i < 5; i++) {
                const card = document.createElement('div');
                card.className = 'preview-card';

                // noisy image
                const canvNoisy = document.createElement('canvas');
                canvNoisy.style.width = '56px'; canvNoisy.style.height = '56px';
                const noisySlice = noisyImgs.slice([i,0,0,0], [1,28,28,1]).squeeze([0]);
                dataLoader.draw28x28ToCanvas(noisySlice, canvNoisy, 2);
                card.appendChild(document.createTextNode('noisy'));
                card.appendChild(canvNoisy);

                // max recon
                const canvMax = document.createElement('canvas');
                canvMax.style.width = '56px'; canvMax.style.height = '56px';
                const maxSlice = reconMax.slice([i,0,0,0], [1,28,28,1]).squeeze([0]);
                dataLoader.draw28x28ToCanvas(maxSlice, canvMax, 2);
                card.appendChild(document.createTextNode('max recon'));
                card.appendChild(canvMax);

                // avg recon
                const canvAvg = document.createElement('canvas');
                canvAvg.style.width = '56px'; canvAvg.style.height = '56px';
                const avgSlice = reconAvg.slice([i,0,0,0], [1,28,28,1]).squeeze([0]);
                dataLoader.draw28x28ToCanvas(avgSlice, canvAvg, 2);
                card.appendChild(document.createTextNode('avg recon'));
                card.appendChild(canvAvg);

                // prediction line
                const predSpan = document.createElement('div');
                predSpan.className = 'prediction';
                const isCorrect = (classPred[i] === trueLabels[i]);
                predSpan.innerHTML = `true: ${trueLabels[i]} / pred: ${classPred[i]}`;
                predSpan.classList.add(isCorrect ? 'correct' : 'wrong');
                card.appendChild(predSpan);

                previewContainer.appendChild(card);

                // dispose slices
                noisySlice.dispose(); maxSlice.dispose(); avgSlice.dispose();
            }

            cleanImgs.dispose(); noisyImgs.dispose(); batch.xs.dispose(); batch.ys.dispose();
            outputs.forEach(t => t.dispose());
            log('🎲 denoising preview ready (max & avg pool variants)');
        } catch (e) {
            log(`❌ test preview error: ${e.message}`);
        }
    });

    // ----- save model (download) -----
    saveModelBtn.addEventListener('click', async () => {
        if (!currentModel) { alert('no model to save'); return; }
        await currentModel.save('downloads://mnist-denoise-cnn');
        log('💾 model saved as download');
    });

    // ----- load model from files (json + bin) -----
    loadModelBtn.addEventListener('click', async () => {
        try {
            const files = loadModelJsonFile.files;
            if (files.length < 2) { alert('select model.json and .bin files'); return; }
            const model = await tf.loadLayersModel(tf.io.browserFiles([files[0], files[1]]));
            if (currentModel) currentModel.dispose();
            currentModel = model;
            currentModel.compile({
                optimizer: 'adam',
                loss: { classifier: 'categoricalCrossentropy', recon_max: 'meanSquaredError', recon_avg: 'meanSquaredError' },
                metrics: { classifier: ['accuracy'] }
            });
            log('📀 model loaded from files');
            modelInfoDiv.innerText = 'loaded custom model (denoising)';
        } catch (e) {
            log(`❌ load model error: ${e.message}`);
        }
    });

    // ----- reset -----
    resetBtn.addEventListener('click', resetAll);

    // ----- toggle tfvis visor -----
    toggleVisorBtn.addEventListener('click', () => tfvis.visor().toggle());

    // initial reset on page load
    resetAll();
})();
