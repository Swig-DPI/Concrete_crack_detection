# (1) Importing dependency
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import save_img
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
import pandas as pd
import time
np.random.seed(1000)


K.tensorflow_backend._get_available_gpus()

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


# dimensions of our images.
img_width, img_height = 256, 256

train_data_dir = 'data/test_train_hold_1/train'
validation_data_dir = 'data/test_train_hold_1/test'
nb_train_samples = 5652
nb_validation_samples = 5652
epochs = 15
batch_size = 50

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)



# (3) Create a sequential model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11),\
 strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# Output Layer of Alexnet
model.add(Dense(17))
model.add(Activation('softmax'))

# model.summary()
#
# # (4) Compile
# model.compile(loss='categorical_crossentropy', optimizer='adam',\
#  metrics=['accuracy'])
#
# # (5) Train
# model.fit(x, y, batch_size=64, epochs=1, verbose=1, \
# validation_split=0.2, shuffle=True)

# my final layer
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])




# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

checkpointer = keras.callbacks.ModelCheckpoint(filepath='weights_white_flip.hdf5', verbose=1, save_best_only=True)
history = LossHistory()
tens_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
## Only run if you change the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks = [checkpointer,history,tens_board])

model.save_weights('Alexnet_trained.h5')
train_generator.reset()
validation_generator.reset()

# add call backs
# keras.callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
# Save the model too

top_model_weights_path = 'Alexnet_trained.h5'
model.load_weights(top_model_weights_path)
scores = model.evaluate_generator(validation_generator, steps = nb_validation_samples // batch_size)
train_generator.reset()
validation_generator.reset()


### train set predictions to CSV
predictions = model.predict_generator(train_generator, steps = nb_train_samples // batch_size)
pred_vals = predictions
vec = np.vectorize(lambda x: 1 if x>0.6 else 0)
predicted_class_indices=vec(predictions)
labels = (validation_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices.ravel()]
filenames=train_generator.filenames[:len(predictions)]
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions,
                      "Values":pred_vals.ravel()})
results.to_csv("Alexnet_trained_results_train.csv",index=False)

train_generator.reset()
validation_generator.reset()
#
# ## Actual predictions to csv
predictions = model.predict_generator(validation_generator, steps = nb_validation_samples // batch_size)
pred_vals = predictions
vec = np.vectorize(lambda x: 1 if x>0.6 else 0)
predicted_class_indices=vec(predictions)
labels = (validation_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices.ravel()]
filenames=validation_generator.filenames[:len(predictions)]
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions,
                      "Values":pred_vals.ravel()})
results.to_csv("Alexnet_trained_results_test.csv",index=False)
