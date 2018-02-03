import os, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def makeDataProcessing(Data):
    listLabel = []
    listImageVector = []
    listImage = []
    for index, strRow in enumerate(Data):
        strLabel, strImageVector = strRow.split(",")
        if index != 0:
            arrayLabel = int(strLabel)
            arrayImageVector = np.fromstring(strImageVector, dtype=int, sep=" ") # for dnn
            arrayImage = arrayImageVector.reshape(48, 48, 1) # for cnn

            listLabel.append(arrayLabel)
            listImageVector.append(arrayImageVector)
            listImage.append(arrayImage)
    return listLabel, listImageVector, listImage


def plot_digits(instances, images_per_row=5, size=140,  **options):
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images)
    plt.imshow(image, cmap="gray", **options)


if __name__ == "__main__":

    strProjectFolder = os.path.dirname(__file__)

    DataTrain = open(os.path.join(strProjectFolder, "01-Data/train.csv"), "r")
    DataTest = open(os.path.join(strProjectFolder, "01-Data/test.csv"), "r")

    listTrainLabel, listTrainImageVector, listTrainImage = makeDataProcessing(DataTrain)
    np.savez(os.path.join(strProjectFolder, "01-Data/Train.npz"), Label=np.asarray(listTrainLabel), Image=np.asarray(listTrainImage))

    _, listTestImageVector, listTestImage = makeDataProcessing(DataTest)
    np.savez(os.path.join(strProjectFolder, "01-Data/Test.npz"), Image=np.asarray(listTestImage))

    listShowId = [0, 299, 2, 7, 3, 15, 4]
    listShowImage = [listTrainImageVector[i] for i in listShowId] 
    plt.figure(figsize=(10, 3))
    plot_digits(instances=listShowImage, images_per_row=7, size=48)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(os.path.dirname(__file__), "02-Output/DisplayData"))
    plt.show()
