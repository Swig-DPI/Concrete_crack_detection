## code help from: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

from __future__ import print_function


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(1337)  # for reproducibility


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K



import numpy as np
import time
from keras.preprocessing.image import save_img


K.tensorflow_backend._get_available_gpus()

# dimensions of our images.
img_width, img_height = 256, 256

train_data_dir = 'data/extra_cracky'
validation_data_dir = 'data/test_train'
nb_train_samples = 1000
nb_validation_samples = 1000
epochs = 25
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(128, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64)) ##  try with Dense(64,kernel_initilizzer = 'glorot_nomral')  might be better
convout1 = Activation('relu')
model.add(convout1)
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
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('second_nn_from_scratch.h5')
train_generator.reset()
validation_generator.reset()

# add call backs
# keras.callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
# Save the model too

top_model_weights_path = 'second_nn_from_scratch.h5'
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
results.to_csv("results_train_gen_second_nn.csv",index=False)

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
results.to_csv("results_validation_gen_second_nn.csv",index=False)

train_generator.reset()
validation_generator.reset()




### visulization of layers




# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)
# layer_name = 'block5_conv1'
layer_name =  model.layers[12].name #'conv2d_7' # conv2d_1'
print(layer_name)

# util function to convert a tensor into a valid image


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# build the VGG16 network with ImageNet weights
# model = vgg16.VGG16(weights='imagenet', include_top=False)  # Model is from above
print('Model loaded.')

model.summary()

# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


kept_filters = []
for filter_index in range(63):
    # we only scan through the first 200 filters,
    # but there are actually 512 of them
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1.

    # we start from a gray image with some random noise
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 3, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # we run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

# we will stich the best 64 filters on a 8 x 8 grid.
n = 2

# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 64 filters.
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

# build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        width_margin = (img_width + margin) * i
        height_margin = (img_height + margin) * j
        stitched_filters[
            width_margin: width_margin + img_width,
            height_margin: height_margin + img_height, :] = img

# save the result to disk
save_img('stitched_filters_%dx%d_second_nn.png' % (n, n), stitched_filters)
