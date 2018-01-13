import os, csv, keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import SGD, Adam


def buildModel(mode):

    model = Sequential()

    if mode == "dnn":
        model.add(Dense(128, input_dim=48*48))
        model.add(Activation("relu"))
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(Dense(7))
        model.add(Activation("softmax"))
        optim = Adam()
    
    if mode == "cnn":
        model.add(Convolution2D(32, kernel_size=3, strides=1, padding="same", input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, kernel_size=3, strides=1, padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dense(32))
        model.add(Activation("relu"))
        model.add(Dense(7))
        model.add(Activation("softmax"))
        optim = Adam()

    model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

if __name__ == "__main__":

    strProjectFolder = os.path.dirname(__file__)

    DataTrain = np.load(os.path.join(strProjectFolder, "01-Data/Train.npz"))

    arrayLabel = keras.utils.to_categorical(DataTrain["Label.npy"])
    arrayImage = DataTrain["Image.npy"]/255.

    model = buildModel(mode="cnn")
    model.fit(arrayImage, arrayLabel, epochs=15, batch_size=128, verbose=2, validation_split=0.25, shuffle=True)



    # listLabel, listImageVector, listImage = makeDataProcessing(DataTest)
    # arrayLabel = keras.utils.to_categorical(listLabel)
    # classes = model.predict(listImageVector, batch_size=128)
