export async function loadCSV(file){

 const text = await file.text()
 const lines = text.split("\n")

 const labels=[]
 const pixels=[]

 for(const line of lines){

  if(!line.trim()) continue

  const parts=line.split(",")

  labels.push(parseInt(parts[0]))

  const row=[]

  for(let i=1;i<=784;i++)
   row.push(Number(parts[i])/255)

  pixels.push(row)
 }

 const xs=tf.tensor2d(pixels).reshape([pixels.length,28,28,1])

 const labelTensor=tf.tensor1d(labels,"int32")

 const ys=tf.oneHot(labelTensor,10)

 labelTensor.dispose()

 return {xs,ys}
}
