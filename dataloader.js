/* =========================================================
DATA LOADER

Reads MNIST CSV files uploaded by the user and converts them
into TensorFlow.js tensors.

CSV format:
label,pixel1,pixel2,...pixel784

All pixels are normalized to [0,1].
========================================================= */

async function parseCSV(file){

  const text = await file.text()
  const lines = text.split('\n')

  const labels=[]
  const pixels=[]

  for(const line of lines){

    if(!line.trim()) continue

    const parts=line.split(',')

    const label=parseInt(parts[0])

    const row=new Float32Array(784)

    for(let i=0;i<784;i++){
      row[i]=Number(parts[i+1])/255
    }

    labels.push(label)
    pixels.push(row)
  }

  return {labels,pixels}
}

function buildTensors(labels,pixels){

  const num=labels.length

  const xs=tf.tensor2d(pixels,[num,784])
    .reshape([num,28,28,1])

  const labelTensor=tf.tensor1d(labels,'int32')
  const ys=tf.oneHot(labelTensor,10)

  labelTensor.dispose()

  return {xs,ys}
}

export async function loadTrainFromFiles(file){

  const {labels,pixels}=await parseCSV(file)
  return buildTensors(labels,pixels)

}

export async function loadTestFromFiles(file){

  const {labels,pixels}=await parseCSV(file)
  return buildTensors(labels,pixels)

}

export function splitTrainVal(xs,ys,valRatio=0.1){

  const size=xs.shape[0]
  const valSize=Math.floor(size*valRatio)
  const trainSize=size-valSize

  const shuffled=tf.util.createShuffledIndices(size)

  const trainIdx=shuffled.slice(0,trainSize)
  const valIdx=shuffled.slice(trainSize)

  const trainXs=tf.gather(xs,trainIdx)
  const trainYs=tf.gather(ys,trainIdx)

  const valXs=tf.gather(xs,valIdx)
  const valYs=tf.gather(ys,valIdx)

  return {trainXs,trainYs,valXs,valYs}
}

export function getRandomTestBatch(xs,ys,k=5){

  const size=xs.shape[0]

  const idx=[]

  for(let i=0;i<k;i++){
    idx.push(Math.floor(Math.random()*size))
  }

  const batchXs=tf.gather(xs,idx)
  const batchYs=tf.gather(ys,idx)

  return {batchXs,batchYs}
}

export function draw28x28ToCanvas(tensor,canvas,scale=4){

  const ctx=canvas.getContext('2d')
  const data=tensor.dataSync()

  canvas.width=28*scale
  canvas.height=28*scale

  for(let y=0;y<28;y++){
    for(let x=0;x<28;x++){

      const val=data[y*28+x]*255
      ctx.fillStyle=`rgb(${val},${val},${val})`
      ctx.fillRect(x*scale,y*scale,scale,scale)

    }
  }
}
