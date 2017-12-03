from help_funcs import *
import csv
import cv2
import numpy as np
np.random.seed(1)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# NN Model
from tensorflow import set_random_seed
set_random_seed(1)
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras import backend as K
K.clear_session()

import matplotlib.pyplot as plt


def MyNet(X_train, y_train, X_valid, y_valid):
    # model = Sequential()
    # # input layer
    # model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(66,200,3)))
    # # Convolutional Layer 1:
    # model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="elu"))
    # # Convolutional Layer 2:
    # model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="elu"))
    # # Convolutional Layer 3:
    # model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="elu"))
    # # Convolutional Layer 4:
    # model.add(Convolution2D(64, 3, 3, activation="elu"))
    # # # Convolutional Layer 5 :
    # model.add(Convolution2D(64, 3, 3, activation="elu"))
    # # Flatten (FC0)
    # model.add(Flatten())
    # # FC1
    # model.add(Dense(100, activation="elu"))
    # # FC2
    # model.add(Dense(50, activation="elu"))
    # # FC3
    # model.add(Dense(10,activation="elu"))
    # # FC4
    # model.add(Dense(1))

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66,200,3)))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))



    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint
    # checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
    #                              monitor='val_loss',
    #                              verbose=0,
    #                              save_best_only=True,
    #                              mode='auto')

    model.compile(loss="mse", optimizer=Adam(lr=0.0001))
    history_object = model.fit(X_train, y_train, batch_size=100, nb_epoch=10,
                               validation_split=0.15, shuffle=True)
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    model.save("model.h5")
    model.summary()
    print("model saved")


lines = open_file("driving_log.csv")
X_train, y_train = get_training_data(lines)
X_valid, y_valid = read_valid("./validation/driving_log.csv")
MyNet(X_train, y_train, X_valid, y_valid)

plt.show()

