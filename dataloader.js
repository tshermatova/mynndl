/* ------------------------------------------------------------------
DATA LOADER MODULE

Responsible for:
1. Reading CSV files from user disk
2. Parsing MNIST rows
3. Converting to tensors
4. Splitting train/validation
5. Sampling test images
6. Rendering 28x28 tensors onto canvas

CSV format (no header):

label,pixel1,pixel2,...pixel784

Example:
5,0,0,0,12,255,...etc

All pixels are normalized to [0,1].

This module avoids external libraries and runs entirely in-browser.
------------------------------------------------------------------ */


/* ---------------------------------------------------------------
CSV PARSER
--------------------------------------------------------------- */

async function parseCSV(file){

  const text = await file.text()

  const lines = text.split('\n')

  const labels = []
  const pixels = []

  for(let line of lines){

    if(!line.trim()) continue

    const parts = line.split(',')

    const label = parseInt(parts[0])

    labels.push(label)

    const row = new Float32Array(784)

    for(let i=0;i<784;i++){
      row[i] = Number(parts[i+1]) / 255
    }

    pixels.push(row)

  }

  return {labels, pixels}
}



/* ---------------------------------------------------------------
TENSOR CREATION
--------------------------------------------------------------- */

function buildTensors(labels, pixels){

  const num = labels.length

  const xs = tf.tensor2d(pixels, [num, 784])
        .reshape([num,28,28,1])

  const labelTensor = tf.tensor1d(labels,'int32')

  const ys = tf.oneHot(labelTensor,10)

  labelTensor.dispose()

  return {xs,ys}
}



/* ---------------------------------------------------------------
PUBLIC FUNCTIONS
--------------------------------------------------------------- */

export async function loadTrainFromFiles(file){

  const {labels,pixels} = await parseCSV(file)

  return buildTensors(labels,pixels)

}

export async function loadTestFromFiles(file){

  const {labels,pixels} = await parseCSV(file)

  return buildTensors(labels,pixels)

}



/* ---------------------------------------------------------------
TRAIN / VALIDATION SPLIT

Random shuffle then split
--------------------------------------------------------------- */

export function splitTrainVal(xs, ys, valRatio=0.1){

  const size = xs.shape[0]

  const valSize = Math.floor(size * valRatio)

  const trainSize = size - valSize

  const shuffled = tf.util.createShuffledIndices(size)

  const trainIdx = shuffled.slice(0,trainSize)
  const valIdx = shuffled.slice(trainSize)

  const trainXs = tf.gather(xs,trainIdx)
  const trainYs = tf.gather(ys,trainIdx)

  const valXs = tf.gather(xs,valIdx)
  const valYs = tf.gather(ys,valIdx)

  return {trainXs,trainYs,valXs,valYs}
}



/* ---------------------------------------------------------------
GET RANDOM TEST BATCH
--------------------------------------------------------------- */

export function getRandomTestBatch(xs,ys,k=5){

  const size = xs.shape[0]

  const indices = []

  for(let i=0;i<k;i++){
    indices.push(Math.floor(Math.random()*size))
  }

  const batchXs = tf.gather(xs,indices)
  const batchYs = tf.gather(ys,indices)

  return {batchXs,batchYs}

}



/* ---------------------------------------------------------------
DRAW 28x28 IMAGE TO CANVAS
--------------------------------------------------------------- */

export function draw28x28ToCanvas(tensor,canvas,scale=4){

  const ctx = canvas.getContext('2d')

  const data = tensor.dataSync()

  const size = 28

  canvas.width = size*scale
  canvas.height = size*scale

  for(let y=0;y<size;y++){

    for(let x=0;x<size;x++){

      const val = data[y*size + x] * 255

      ctx.fillStyle = `rgb(${val},${val},${val})`

      ctx.fillRect(x*scale,y*scale,scale,scale)

    }

  }

}
