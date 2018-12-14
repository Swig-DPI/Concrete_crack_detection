import keras
from keras.applications import InceptionV3, Xception, VGG16
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import RMSprop, adadelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from simple_cnn import create_cnn
import os
import glob
from Making_the_model import create_transfer_model

# def channel_to_three(img):
#     return np.stack((img,)*3, axis=-1)

class ClassificationCNN():

    def __init__(self, project_name, target_size, channels=1, augmentation_strength=0.1, preprocessing=None, batch_size=50, scale=255):
        self.project_name = project_name
        self.target_size = target_size
        if channels == 3:
            self.input_size = self.target_size + (3,)
            self.color_mode = 'rgb'
        else:
            self.input_size = self.target_size + (1,)
            self.color_mode = 'grayscale'
        self.train_datagen = None
        self.train_generator= None
        self.validation_datagen = None
        self.validation_generator = None
        self.holdout_generator = None
        self.augmentation_strength = augmentation_strength
        self.preprocessing = preprocessing
        self.batch_size = batch_size
        self.scale = scale #scale is included for to use either uint8 or uint16 images
        self.class_names =  None

        self.loss_function = None
        self.class_mode = None
        self.last_activation = None
        self.history = None

    def get_data(self, train_folder, validation_folder, holdout_folder):

        self.train_folder = train_folder
        self.validation_folder = validation_folder
        self.holdout_folder = holdout_folder

        self.num_train = sum(len(files) for _, _, files in os.walk(self.train_folder)) #: number of training samples

        self.num_val = sum(len(files) for _, _, files in os.walk(self.validation_folder)) #: number of validation samples

        self.num_holdout = sum(len(files) for _, _, files in os.walk(self.holdout_folder)) #: number of holdout samples

        self.num_categories = sum(len(dirnames) for _, dirnames, _ in os.walk(self.train_folder)) #: number of categories

        # self.class_names = self.set_class_names() #: text representation of classes


    def build_generators(self):
        '''Create generators to read images from directory'''

        self.train_datagen = ImageDataGenerator(
                        preprocessing_function = self.preprocessing,
                        rescale=1./self.scale,
                        rotation_range=self.augmentation_strength,
                        width_shift_range=self.augmentation_strength,
                        height_shift_range=self.augmentation_strength,
                        shear_range=self.augmentation_strength,
                        zoom_range=self.augmentation_strength,
                        horizontal_flip=True)

        self.validation_datagen = ImageDataGenerator(
                        preprocessing_function = self.preprocessing,
                        rescale=1./self.scale)

        self.train_generator = self.train_datagen.flow_from_directory(
                            self.train_folder,
                            color_mode=self.color_mode,
                            target_size=self.target_size,
                            batch_size=self.batch_size,
                            class_mode=self.class_mode,
                            shuffle=True)

        self.validation_generator = self.validation_datagen.flow_from_directory(
                            self.validation_folder,
                            color_mode=self.color_mode,
                            target_size=self.target_size,
                            batch_size=self.batch_size,
                            class_mode=self.class_mode,
                            shuffle=True)

    def fit(self, input_model, train_folder, validation_folder, holdout_folder, epochs, loss, optimizer='adadelta'):
        ''' 1) Build all generators with flow from directory
            2) Build simple cnn with the option of changing last layer to softmax (more that one class)
                and sigmoid (only one class)
            3)'''

        self.loss_function = loss
        if self.loss_function == 'categorical_crossentropy':
            self.class_mode = 'categorical'
            self.last_activation = 'softmax'
        elif self.loss_function == 'binary_crossentropy':
            self.class_mode = 'binary'
            self.last_activation = 'sigmoid'
        else:
            print('WARNING: Please specify loss function as categorical or binary crossentropy')

        self.get_data(train_folder, validation_folder, holdout_folder)
        print(self.class_names, self.loss_function)
        self.build_generators()
        model = input_model(self.input_size, loss)

        model.compile(optimizer=optimizer, loss=self.loss_function, metrics=['accuracy'])

        #initialize tensorboard for monitoring
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=self.project_name, histogram_freq=0, batch_size=self.batch_size, write_graph=True, embeddings_freq=0)

        #initialize model checkpoint to save the best model
        save_name = 'save_model/'+self.project_name+'.hdf5'
        call_backs = [ModelCheckpoint(filepath=save_name,
                                    monitor='val_loss',
                                    save_best_only=True),
                                    EarlyStopping(monitor='val_loss', patience=5, verbose=0)]

        self.history = model.fit_generator(
                self.train_generator,
                steps_per_epoch=self.num_train// self.batch_size,
                epochs=epochs,
                validation_data=self.validation_generator,
                validation_steps=self.num_val // self.batch_size,
                callbacks=call_backs)

        best_model = load_model(save_name)
        print('evaluating simple model')
        accuracy = self.evaluate_model(best_model, self.holdout_folder)

        return save_name

    def evaluate_model(self, model, holdout_folder):
        """
        evaluates model on holdout data
        Args:
            model (keras classifier model): model to evaluate
            holdout_folder (str): path of holdout data
        Returns:
            list(float): metrics returned by the model, typically [loss, accuracy]
            """

        self.holdout_generator = self.validation_datagen.flow_from_directory(
                            self.holdout_folder,
                            color_mode=self.color_mode,
                            target_size=self.target_size,
                            batch_size=self.batch_size,
                            class_mode=self.class_mode,
                            shuffle=False)


        metrics = model.evaluate_generator(self.holdout_generator,
                                           steps=self.num_holdout/self.batch_size,
                                           use_multiprocessing=True,
                                           verbose=1)
        print(f"holdout loss: {metrics[0]} accuracy: {metrics[1]}")
        return metrics


    def set_class_names(self):
        """
        Sets the class names, sorted by alphabetical order
        """
        names = [os.path.basename(x) for x in glob(self.train_folder + '/*')]

        return sorted(names)


    def plot_model(self):
        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(1, len(acc) +1)

        plt.plot(epochs, acc, 'g-', label='Training acc')
        plt.plot(epochs, val_acc, 'b-', label='Validation acc')
        plt.title('Training and validation accuracy', fontsize=18)
        plt.xlabel('epochs', fontsize=16)
        plt.ylabel('loss', fontsize=16)
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'g-', label='Training loss')
        plt.plot(epochs, val_loss, 'b-', label='Validation loss')
        plt.title('Training and validation loss', fontsize=18)
        plt.xlabel('epochs', fontsize=16)
        plt.ylabel('loss', fontsize=16)
        plt.legend()
        plt.show()

    def plot_model_2(self):
        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(1, len(acc) +1)

        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9,5))
        ax0.plot(epochs, acc, 'g-', label='Training acc')
        ax0.plot(epochs, val_acc, 'b-', label='Validation acc')
        ax0.set_title('Training and validation accuracy', fontsize=16)
        ax0.set_xlabel('epochs', fontsize=14)
        ax0.set_ylabel('loss', fontsize=14)
        ax0.legend()

        ax1.plot(epochs, loss, 'g-', label='Training loss')
        ax1.plot(epochs, val_loss, 'b-', label='Validation loss')
        ax1.set_title('Training and validation loss', fontsize=16)
        ax1.set_xlabel('epochs', fontsize=14)
        ax1.set_ylabel('loss', fontsize=14)
        ax1.legend()
        plt.tight_layout()
        plt.show()

class TransferCNN(ClassificationCNN):

    def fit(self, model_name, train_folder, validation_folder, holdout_folder, input_model, n_categories, loss, optimizers, epochs, freeze_indices, warmup_epochs=5):

        self.n_categories = n_categories

        self.loss_function = loss
        if self.loss_function == 'categorical_crossentropy':
            self.class_mode = 'categorical'
            self.last_activation = 'softmax'
        elif self.loss_function == 'binary_crossentropy':
            self.class_mode = 'binary'
            self.last_activation = 'sigmoid'
        else:
            print('WARNING: Please specify loss function as categorical or binary crossentropy')

        self.get_data(train_folder, validation_folder, holdout_folder)
        self.build_generators()

        model = input_model(self.input_size, self.n_categories, self.last_activation, model=model_name)
        self.change_trainable_layers(model, freeze_indices[0])

        model.compile(optimizer=optimizers[0],
                        loss=self.loss_function, metrics=['accuracy'])

        tensorboard = keras.callbacks.TensorBoard(
            log_dir=self.project_name, histogram_freq=0, batch_size=self.batch_size, write_graph=True, embeddings_freq=0)

        save_name = 'save_model/'+self.project_name+'.hdf5'
        call_backs = [ModelCheckpoint(filepath=save_name,
                                    monitor='val_loss',
                                    save_best_only=True),
                                    EarlyStopping(monitor='val_loss', patience=5, verbose=0)]

        self.history = model.fit_generator(
                self.train_generator,
                steps_per_epoch=self.num_train// self.batch_size,
                epochs=warmup_epochs,
                validation_data=self.validation_generator,
                validation_steps=self.num_val // self.batch_size,
                callbacks=call_backs)

        self.change_trainable_layers(model, freeze_indices[1])

        model.compile(optimizer=optimizers[1],
                      loss=self.loss_function, metrics=['accuracy'])

        self.history = model.fit_generator(
                self.train_generator,
                steps_per_epoch=self.num_train// self.batch_size,
                epochs=epochs,
                validation_data=self.validation_generator,
                validation_steps=self.num_val // self.batch_size,
                callbacks=call_backs)

        if len(freeze_indices) > 2:
            for i in range(len(freeze_indices) - 2):
                i = i + 2
                self.change_trainable_layers(model, freeze_indices[i])

                model.compile(optimizer=optimizers[1],
                              loss=self.loss_function, metrics=['accuracy'])

                self.history = model.fit_generator(
                        self.train_generator,
                        steps_per_epoch=self.num_train// self.batch_size,
                        epochs=epochs,
                        validation_data=self.validation_generator,
                        validation_steps=self.num_val // self.batch_size,
                        callbacks=call_backs)


        best_model = load_model(save_name)
        print('evaluating transfer model')
        accuracy = self.evaluate_model(best_model, self.holdout_folder)
        print('evaluating transfer model complete')

        return save_name


    def change_trainable_layers(self, model, trainable_index, freeze_indices_gt1=False):

        if freeze_indices_gt1 == False:
            for layer in model.layers[:trainable_index]:
                layer.trainable = False
            for layer in model.layers[trainable_index:]:
                layer.trainable = True
        else:
            for layer in model.layers[:trainable_index]:
                layer.trainable = False
            for layer in model.layers[trainable_index:]:
                layer.trainable = True


    def ml_classifier():
        '''feature extraction to ml model'''


if __name__ == '__main__':

    train_folder = '../data/test_train_hold_1/train'
    validation_folder = '../data/test_train_hold_1/test'
    holdout_folder = '../data/test_train_hold_1/hold'

    #simple cnn
    input_shape = (256,256,3)
    target_size = (256,256)
    scale = 255
    epochs = 30

    # simple_model = create_cnn
    # simple_cnn = ClassificationCNN('class_test_one', target_size, channels=1, augmentation_strength=0.1, preprocessing=None, batch_size=50, scale=65535)
    # simple_cnn.fit(simple_model, train_folder, validation_folder, holdout_folder, epochs, 'categorical_crossentropy', optimizer='adadelta')

    # inception
    # model_name = InceptionV3
    # warmup_epochs = 3
    # epochs = epochs - warmup_epochs
    # optimizers = [RMSprop(lr=0.0006), RMSprop(lr=0.0001)] # keep learning rates low to keep from wrecking weights
    # train_head_idx = [311, 299]
    # transfer_model = create_transfer_model
    # transfer_cnn = TransferCNN('transfer_test_one', target_size, channels=3, augmentation_strength=0.1, preprocessing=None, batch_size=50, scale=255)
    # savename = transfer_cnn.fit(train_folder, validation_folder, holdout_folder, transfer_model, 2, \
    #                     'categorical_crossentropy', optimizers, epochs, train_head_idx, warmup_epochs=warmup_epochs)

    #xception
    # model_name = Xception
    # warmup_epochs = 1
    # epochs = epochs - warmup_epochs
    # optimizers = [RMSprop(lr=0.0006), RMSprop(lr=0.0001)] # keep learning rates low to keep from wrecking weights
    # train_head_idx = [132, 126]
    # transfer_model = create_transfer_model
    # transfer_cnn = TransferCNN('transfer_test_one', target_size, channels=3, augmentation_strength=0.1, preprocessing=None, batch_size=50, scale=255)
    # savename = transfer_cnn.fit(model_name, train_folder, validation_folder, holdout_folder, transfer_model, 2, \
    #                     'categorical_crossentropy', optimizers, epochs, train_head_idx, warmup_epochs=warmup_epochs)

    #VGG16
    model_name = VGG16
    warmup_epochs = 10
    epochs = epochs - warmup_epochs
    optimizers = [RMSprop(lr=0.0006), RMSprop(lr=0.0001)] # keep learning rates low to keep from wrecking weights
    train_head_idx = [19, 17, 15]
    transfer_model = create_transfer_model
    transfer_cnn = TransferCNN('transfer_test_one', target_size, channels=3, augmentation_strength=0.1, preprocessing=None, batch_size=50, scale=255)
    print('Begin fit Model')
    savename = transfer_cnn.fit(model_name, train_folder, validation_folder, holdout_folder, transfer_model, 2, \
                        'binary_crossentropy', optimizers, epochs, train_head_idx, warmup_epochs=warmup_epochs)
    print('Model Fit complete')
