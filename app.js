/* ----------------------------------------------------------------
MAIN APPLICATION CONTROLLER

Handles:

UI events
Model creation
Training
Evaluation
Preview testing
Saving/loading models

This file intentionally includes heavy comments to help students
understand how a full ML training pipeline works in the browser.
---------------------------------------------------------------- */


import {
loadTrainFromFiles,
loadTestFromFiles,
splitTrainVal,
getRandomTestBatch,
draw28x28ToCanvas
} from './data-loader.js'



/* --------------------------------------------------------------
GLOBAL STATE
-------------------------------------------------------------- */

let trainXs, trainYs
let testXs, testYs

let model = null



/* --------------------------------------------------------------
UI HELPERS
-------------------------------------------------------------- */

function log(msg){

  const el = document.getElementById("log")

  el.textContent += msg + "\n"

  el.scrollTop = el.scrollHeight

}



/* --------------------------------------------------------------
MODEL CREATION

Classic CNN for MNIST classification.
-------------------------------------------------------------- */

function buildModel(){

  const m = tf.sequential({

    layers:[

      tf.layers.conv2d({
        filters:32,
        kernelSize:3,
        activation:'relu',
        padding:'same',
        inputShape:[28,28,1]
      }),

      tf.layers.conv2d({
        filters:64,
        kernelSize:3,
        activation:'relu',
        padding:'same'
      }),

      tf.layers.maxPooling2d({poolSize:2}),

      tf.layers.dropout({rate:0.25}),

      tf.layers.flatten(),

      tf.layers.dense({
        units:128,
        activation:'relu'
      }),

      tf.layers.dropout({rate:0.5}),

      tf.layers.dense({
        units:10,
        activation:'softmax'
      })

    ]

  })

  m.compile({
    optimizer:'adam',
    loss:'categoricalCrossentropy',
    metrics:['accuracy']
  })

  return m

}



/* --------------------------------------------------------------
LOAD DATA
-------------------------------------------------------------- */

async function onLoadData(){

  try{

    const trainFile = document.getElementById("trainFile").files[0]
    const testFile = document.getElementById("testFile").files[0]

    log("Reading CSV files...")

    const train = await loadTrainFromFiles(trainFile)
    const test = await loadTestFromFiles(testFile)

    trainXs = train.xs
    trainYs = train.ys

    testXs = test.xs
    testYs = test.ys

    document.getElementById("dataStatus").innerText =
      `Train: ${trainXs.shape[0]} samples | Test: ${testXs.shape[0]}`

    log("Data loaded successfully.")

  }catch(e){

    log("Error loading data: " + e)

  }

}



/* --------------------------------------------------------------
TRAIN MODEL
-------------------------------------------------------------- */

async function onTrain(){

  if(!trainXs){
    log("Load data first.")
    return
  }

  if(model) model.dispose()

  model = buildModel()

  const {trainXs:tx,trainYs:ty,valXs,v alYs} = splitTrainVal(trainXs,trainYs)

  document.getElementById("modelInfo").textContent = model.summary()

  const visor = tfvis.visor()

  visor.open()

  const metrics = ['loss','val_loss','acc','val_acc']

  const surface = { name: 'Training', tab: 'Training'}

  const callbacks = tfvis.show.fitCallbacks(surface,metrics)

  log("Training started...")

  await model.fit(tx,ty,{

    epochs:8,
    batchSize:128,
    shuffle:true,
    validationData:[valXs,valYs],
    callbacks

  })

  log("Training complete.")

}



/* --------------------------------------------------------------
EVALUATE MODEL
-------------------------------------------------------------- */

async function onEvaluate(){

  if(!model || !testXs) return

  const preds = model.predict(testXs).argMax(-1)

  const labels = testYs.argMax(-1)

  const predsData = await preds.data()
  const labelsData = await labels.data()

  let correct = 0

  for(let i=0;i<predsData.length;i++){
    if(predsData[i]===labelsData[i]) correct++
  }

  const acc = correct/predsData.length

  document.getElementById("metrics").innerText =
    `Accuracy: ${(acc*100).toFixed(2)}%`

  tfvis.metrics.confusionMatrix(
    {name:"Confusion Matrix"},
    labelsData,
    predsData
  )

}



/* --------------------------------------------------------------
TEST 5 RANDOM SAMPLES
-------------------------------------------------------------- */

async function onTestFive(){

  const {batchXs,batchYs} = getRandomTestBatch(testXs,testYs,5)

  const preds = model.predict(batchXs).argMax(-1)

  const predVals = await preds.data()

  const labelVals = await batchYs.argMax(-1).data()

  const row = document.getElementById("previewRow")

  row.innerHTML = ""

  for(let i=0;i<5;i++){

    const item = document.createElement("div")
    item.className = "previewItem"

    const canvas = document.createElement("canvas")

    draw28x28ToCanvas(batchXs.slice([i,0,0,0],[1,28,28,1]).reshape([28,28]),canvas)

    const p = document.createElement("div")

    const correct = predVals[i]===labelVals[i]

    p.textContent = `Pred: ${predVals[i]} | True: ${labelVals[i]}`

    p.className = correct ? "correct" : "wrong"

    item.appendChild(canvas)
    item.appendChild(p)

    row.appendChild(item)

  }

}



/* --------------------------------------------------------------
SAVE MODEL
-------------------------------------------------------------- */

async function onSave(){

  if(model) await model.save('downloads://mnist-cnn')

}



/* --------------------------------------------------------------
LOAD MODEL FROM FILES
-------------------------------------------------------------- */

async function onLoadModel(){

  const json = document.getElementById("modelJson").files[0]
  const bin = document.getElementById("modelBin").files[0]

  model = await tf.loadLayersModel(tf.io.browserFiles([json,bin]))

  model.summary()

  log("Model loaded.")

}



/* --------------------------------------------------------------
RESET APP
-------------------------------------------------------------- */

function onReset(){

  if(model) model.dispose()

  trainXs?.dispose()
  trainYs?.dispose()
  testXs?.dispose()
  testYs?.dispose()

  document.getElementById("previewRow").innerHTML=""
  document.getElementById("log").textContent=""
  document.getElementById("metrics").textContent=""

}



/* --------------------------------------------------------------
VISOR TOGGLE
-------------------------------------------------------------- */

function onToggleVisor(){

  tfvis.visor().toggle()

}



/* --------------------------------------------------------------
EVENT BINDING
-------------------------------------------------------------- */

document.getElementById("loadBtn").onclick = onLoadData
document.getElementById("trainBtn").onclick = onTrain
document.getElementById("evalBtn").onclick = onEvaluate
document.getElementById("testFiveBtn").onclick = onTestFive
document.getElementById("saveBtn").onclick = onSave
document.getElementById("loadModelBtn").onclick = onLoadModel
document.getElementById("resetBtn").onclick = onReset
document.getElementById("toggleVisorBtn").onclick = onToggleVisor
