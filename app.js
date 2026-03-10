async function train(){

 const history = await model.fit(trainXs,trainYs,{

  epochs:8,
  batchSize:128,
  shuffle:true,
  validationSplit:0.1,

  callbacks: tfvis.show.fitCallbacks(
   {name:"Training"},
   ["loss","val_loss","acc","val_acc"]
  )

 })

}
