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
        # model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Convolution2D(64, kernel_size=3, strides=1, padding="same"))
        # model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Convolution2D(128, kernel_size=3, strides=1, padding="same"))
        # model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

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
    # keras.utils.plot_model(model, to_file=os.path.join(strProjectFolder, "02-Output/model.png"), show_shapes=True)
    Training = model.fit(arrayImage, arrayLabel, epochs=15, batch_size=128, verbose=2, validation_split=0.25, shuffle=True)

    # model.save(os.path.join(strProjectFolder, "02-Output/model"))

    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(Training.history["loss"], "r", linewidth=3.0)
    plt.plot(Training.history["val_loss"], "b", linewidth=3.0)
    plt.legend(["Training loss", "Validation Loss"], fontsize=18)
    plt.xlabel("Epochs ", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.title("Loss Curves", fontsize=16)
    
    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(Training.history["acc"], "r", linewidth=3.0)
    plt.plot(Training.history["val_acc"], "b", linewidth=3.0)
    plt.legend(["Training Accuracy", "Validation Accuracy"], fontsize=18)
    plt.xlabel("Epochs ", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.title("Accuracy Curves", fontsize=16)
    plt.show()



    # listLabel, listImageVector, listImage = makeDataProcessing(DataTest)
    # classes = model.predict(listImageVector, batch_size=128)


