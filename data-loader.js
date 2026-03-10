/* data-loader.js – CSV parsing, tensor preparation, noise addition, denoising utils */

// ---------- core CSV parsing (streaming-friendly but readAsText for simplicity) ----------
async function parseCSVFromFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const text = e.target.result;
                const lines = text.split(/\r?\n/).filter(line => line.trim() !== '');
                const rows = [];
                for (let line of lines) {
                    const values = line.split(',').map(v => {
                        const num = parseFloat(v);
                        return isNaN(num) ? 0 : num;
                    });
                    if (values.length !== 785) continue; // malformed skip
                    rows.push(values);
                }
                resolve(rows);
            } catch (err) {
                reject(err);
            }
        };
        reader.onerror = () => reject(reader.error);
        reader.readAsText(file); // using readAsText for clarity, large files ok for MNIST
    });
}

// normalize [0,255] -> [0,1], reshape to [N,28,28,1], one-hot labels depth 10
function rowsToTensors(rows) {
    return tf.tidy(() => {
        if (!rows.length) throw new Error('empty rows');
        const numExamples = rows.length;
        const labels = new Array(numExamples);
        const images = new Array(numExamples);
        rows.forEach((row, idx) => {
            labels[idx] = row[0];               // first element = label
            images[idx] = row.slice(1).map(v => v / 255.0); // normalize to 0-1
        });

        // create one-hot labels
        const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 10);

        // images flat array → reshape
        const xs = tf.tensor2d(images, [numExamples, 784]).reshape([numExamples, 28, 28, 1]);

        return { xs, ys };
    });
}

// public: load train file
async function loadTrainFromFiles(file) {
    const rows = await parseCSVFromFile(file);
    return rowsToTensors(rows);
}

// public: load test file
async function loadTestFromFiles(file) {
    const rows = await parseCSVFromFile(file);
    return rowsToTensors(rows);
}

// split train into train/val (10% validation)
function splitTrainVal(xs, ys, valRatio = 0.1) {
    return tf.tidy(() => {
        const numTrain = xs.shape[0];
        const numVal = Math.floor(numTrain * valRatio);
        const numTrainMain = numTrain - numVal;

        const trainXs = xs.slice([0, 0, 0, 0], [numTrainMain, 28, 28, 1]);
        const trainYs = ys.slice([0, 0], [numTrainMain, 10]);

        const valXs = xs.slice([numTrainMain, 0, 0, 0], [numVal, 28, 28, 1]);
        const valYs = ys.slice([numTrainMain, 0], [numVal, 10]);

        return { trainXs, trainYs, valXs, valYs };
    });
}

// get random batch of k test images (original, no noise)
function getRandomTestBatch(xs, ys, k = 5) {
    return tf.tidy(() => {
        const total = xs.shape[0];
        const indices = [];
        for (let i = 0; i < k; i++) {
            indices.push(Math.floor(Math.random() * total));
        }
        const batchXs = tf.gather(xs, tf.tensor1d(indices, 'int32'));
        const batchYs = tf.gather(ys, tf.tensor1d(indices, 'int32'));
        return { xs: batchXs, ys: batchYs };
    });
}

// ---------- denoising utils (add noise, autoencoder inference) ----------
// add random uniform noise [0, maxNoise] to test images (clipped to [0,1])
function addNoiseToTensors(xs, noiseLevel = 0.3) {
    return tf.tidy(() => {
        const noise = tf.randomUniform(xs.shape, 0, noiseLevel);
        const noisy = xs.add(noise);
        return noisy.clipByValue(0, 1);
    });
}

// apply denoising autoencoder model (maxpool & avgpool variants returned)
// returns two denoised tensors: one from maxpool path, one from avgpool path
// we assume model has two outputs: [classifier_output, maxpool_recon, avgpool_recon]
function denoiseWithModel(model, noisyXs) {
    return tf.tidy(() => {
        // we need to run the whole model, but we only have classifier head normally.
        // Instead, we split the model: encoder part (conv+pool) and decoder branches.
        // For simplicity in this demo, we build a separate denoising model in app.js
        // But here we provide utility to extract reconstructions if model has multiple outputs.
        // We will implement inside app.js training a multi-output model.
        // So this function is kept as placeholder; actual logic in app.js.
        console.warn('data-loader: denoiseWithModel called, but implemented in app.js');
        return { maxpoolDenoised: noisyXs, avgpoolDenoised: noisyXs }; // fallback
    });
}

// draw 28x28 tensor to canvas (scale factor)
function draw28x28ToCanvas(tensor, canvas, scale = 4) {
    return tf.tidy(() => {
        const data = tensor.squeeze().arraySync(); // [28,28]
        const ctx = canvas.getContext('2d');
        const width = 28 * scale, height = 28 * scale;
        canvas.width = width; canvas.height = height;
        const imgData = ctx.createImageData(width, height);
        for (let i = 0; i < 28; i++) {
            for (let j = 0; j < 28; j++) {
                const val = Math.floor(data[i][j] * 255);
                const color = (val << 16) | (val << 8) | val | 0xff000000;
                for (let dx = 0; dx < scale; dx++) {
                    for (let dy = 0; dy < scale; dy++) {
                        const px = (i * scale + dx) * width + (j * scale + dy);
                        imgData.data[px * 4] = val;
                        imgData.data[px * 4 + 1] = val;
                        imgData.data[px * 4 + 2] = val;
                        imgData.data[px * 4 + 3] = 255;
                    }
                }
            }
        }
        ctx.putImageData(imgData, 0, 0);
    });
}

// export public interface
window.dataLoader = {
    loadTrainFromFiles,
    loadTestFromFiles,
    splitTrainVal,
    getRandomTestBatch,
    draw28x28ToCanvas,
    addNoiseToTensors,
    denoiseWithModel
};
