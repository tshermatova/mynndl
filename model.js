export function createModel(){

 const model=tf.sequential()

 model.add(tf.layers.conv2d({
  inputShape:[28,28,1],
  filters:32,
  kernelSize:3,
  activation:"relu",
  padding:"same"
 }))

 model.add(tf.layers.conv2d({
  filters:64,
  kernelSize:3,
  activation:"relu",
  padding:"same"
 }))

 model.add(tf.layers.maxPooling2d({poolSize:2}))

 model.add(tf.layers.dropout({rate:0.25}))

 model.add(tf.layers.flatten())

 model.add(tf.layers.dense({
  units:128,
  activation:"relu"
 }))

 model.add(tf.layers.dropout({rate:0.5}))

 model.add(tf.layers.dense({
  units:10,
  activation:"softmax"
 }))

 model.compile({
  optimizer:"adam",
  loss:"categoricalCrossentropy",
  metrics:["accuracy"]
 })

 return model
}
