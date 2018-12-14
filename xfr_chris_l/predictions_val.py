
top_model_weights_path = savename
model = load_weights(top_model_weights_path)
scores = model.evaluate_generator(transfer_cnn.holdout_generator,
                                   steps=transfer_cnn.num_holdout/transfer_cnn.batch_size,
                                   use_multiprocessing=True,
                                   verbose=1)
print(f"holdout loss: {metrics[0]} accuracy: {metrics[1]}")
train_generator.reset()
validation_generator.reset()

### train set predictions to CSV
predictions = model.predict_generator(self.holdout_generator, steps = transfer_cnn.num_holdout/transfer_cnn.batch_size)
pred_vals = predictions
vec = np.vectorize(lambda x: 1 if x>0.6 else 0)
predicted_class_indices=vec(predictions)
labels = (transfer_cnn.holdout_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices.ravel()]
filenames=transfer_cnn.holdout_generator.filenames[:len(predictions)]
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions,
                      "Values":pred_vals.ravel()})
results.to_csv("transfer_learning_holdout_set.csv",index=False)
