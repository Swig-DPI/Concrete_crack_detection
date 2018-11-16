## code help from: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(1337)  # for reproducibility


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# dimensions of our images.
img_width, img_height = 256, 256

train_data_dir = 'data/extra_cracky'
validation_data_dir = 'data/test_train'
nb_train_samples = 1000
nb_validation_samples = 1000
epochs = 12
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
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


## Only run if you change the model
# model.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size)
#
# model.save_weights('second_try.h5')
# train_generator.reset()
# validation_generator.reset()

# add call backs
# keras.callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
# Save the model too

top_model_weights_path = 'second_try.h5'
model.load_weights(top_model_weights_path)
scores = model.evaluate_generator(validation_generator, steps = nb_validation_samples // batch_size)
train_generator.reset()
validation_generator.reset()


### train set predictions to CSV
predictions = model.predict_generator(train_generator, steps = nb_train_samples // batch_size)
predicted_class_indices=np.argmax(predictions,axis=1)
labels = (validation_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames=train_generator.filenames[:len(predictions)]
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results_train_gen.csv",index=False)

train_generator.reset()
validation_generator.reset()

## Actual predictions to csv
predictions = model.predict_generator(validation_generator, steps = nb_validation_samples // batch_size)
predicted_class_indices=np.argmax(predictions,axis=1)
labels = (validation_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames=validation_generator.filenames[:len(predictions)]
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results_validation_gen.csv",index=False)

train_generator.reset()
validation_generator.reset()

# x,y = validation_generator.next()
# for i in range(0,1):
#     image = x[i]
#     plt.imshow(image)
#     plt.show()
