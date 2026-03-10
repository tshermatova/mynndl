import {
loadTrainFromFiles,
loadTestFromFiles,
splitTrainVal,
getRandomTestBatch,
draw28x28ToCanvas
} from './data-loader.js'

let trainXs,trainYs
let testXs,testYs
let model=null

function log(msg){
  const el=document.getElementById("log")
  el.textContent+=msg+"\n"
  el.scrollTop=el.scrollHeight
}

function buildModel(){

  const m=tf.sequential({

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

      tf.layers.dense({units:128,activation:'relu'}),
      tf.layers.dropout({rate:0.5}),

      tf.layers.dense({units:10,activation:'softmax'})
    ]

  })

  m.compile({
    optimizer:'adam',
    loss:'categoricalCrossentropy',
    metrics:['accuracy']
  })

  return m
}

async function onLoadData(){

  try{

    const trainFile=document.getElementById("trainFile").files[0]
    const testFile=document.getElementById("testFile").files[0]

    if(!trainFile||!testFile){
      alert("Please upload both train and test CSV files")
      return
    }

    log("Loading training data...")
    const train=await loadTrainFromFiles(trainFile)

    log("Loading test data...")
    const test=await loadTestFromFiles(testFile)

    trainXs=train.xs
    trainYs=train.ys
    testXs=test.xs
    testYs=test.ys

    document.getElementById("dataStatus").innerText=
      `Train: ${trainXs.shape[0]} | Test: ${testXs.shape[0]}`

    log("Data loaded successfully")

  }catch(e){
    console.error(e)
    log("Error loading data: "+e.message)
  }
}

async function onTrain(){

  if(!trainXs){
    log("Load data first")
    return
  }

  if(model) model.dispose()

  model=buildModel()

  const {trainXs:tx,trainYs:ty,valXs,valYs}=splitTrainVal(trainXs,trainYs)

  log("Training started")

  await model.fit(tx,ty,{
    epochs:8,
    batchSize:128,
    shuffle:true,
    validationData:[valXs,valYs],
    callbacks:tfvis.show.fitCallbacks(
      {name:"Training"},
      ["loss","val_loss","acc","val_acc"]
    )
  })

  log("Training finished")
}

async function onEvaluate(){

  if(!model||!testXs){
    log("Model or data missing")
    return
  }

  const preds=model.predict(testXs).argMax(-1)
  const labels=testYs.argMax(-1)

  const p=await preds.data()
  const l=await labels.data()

  let correct=0

  for(let i=0;i<p.length;i++){
    if(p[i]===l[i]) correct++
  }

  const acc=correct/p.length

  document.getElementById("metrics").innerText=
    `Accuracy ${(acc*100).toFixed(2)}%`

}

async function onTestFive(){

  const {batchXs,batchYs}=getRandomTestBatch(testXs,testYs,5)

  const preds=model.predict(batchXs).argMax(-1)

  const p=await preds.data()
  const l=await batchYs.argMax(-1).data()

  const row=document.getElementById("previewRow")
  row.innerHTML=""

  for(let i=0;i<5;i++){

    const item=document.createElement("div")
    item.className="previewItem"

    const canvas=document.createElement("canvas")

    draw28x28ToCanvas(
      batchXs.slice([i,0,0,0],[1,28,28,1]).reshape([28,28]),
      canvas
    )

    const label=document.createElement("div")
    label.textContent=`Pred ${p[i]} / True ${l[i]}`

    label.className=p[i]===l[i]?"correct":"wrong"

    item.appendChild(canvas)
    item.appendChild(label)

    row.appendChild(item)
  }
}

async function onSave(){
  if(model) await model.save("downloads://mnist-cnn")
}

async function onLoadModel(){

  const json=document.getElementById("modelJson").files[0]
  const bin=document.getElementById("modelBin").files[0]

  model=await tf.loadLayersModel(
    tf.io.browserFiles([json,bin])
  )

  log("Model loaded")
}

function onReset(){

  if(model){
    model.dispose()
    model=null
  }

  trainXs?.dispose()
  trainYs?.dispose()
  testXs?.dispose()
  testYs?.dispose()

  trainXs=trainYs=testXs=testYs=null

  document.getElementById("trainFile").value=""
  document.getElementById("testFile").value=""

  document.getElementById("previewRow").innerHTML=""
  document.getElementById("metrics").innerHTML=""
  document.getElementById("log").textContent=""

  document.getElementById("dataStatus").textContent="No data loaded"

  log("App reset complete")
}

function onToggleVisor(){
  tfvis.visor().toggle()
}

document.getElementById("loadBtn").onclick=onLoadData
document.getElementById("trainBtn").onclick=onTrain
document.getElementById("evalBtn").onclick=onEvaluate
document.getElementById("testFiveBtn").onclick=onTestFive
document.getElementById("saveBtn").onclick=onSave
document.getElementById("loadModelBtn").onclick=onLoadModel
document.getElementById("resetBtn").onclick=onReset
document.getElementById("toggleVisorBtn").onclick=onToggleVisor
