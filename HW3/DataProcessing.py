import os, csv
import numpy as np


def makeDataProcessing(Data):
    listLabel = []
    listImageVector = []
    listImage = []
    for index, strRow in enumerate(Data):
        strLabel, strImageVector = strRow.split(",")
        if index != 0:
            arrayLabel = [int(strLabel)]
            arrayImageVector = np.fromstring(strImageVector, dtype=int, sep=" ") # for dnn
            arrayImage = arrayImageVector.reshape(48, 48, 1) # for cnn

            listLabel.append(arrayLabel)
            listImageVector.append(arrayImageVector)
            listImage.append(arrayImage)
    return listLabel, listImageVector, listImage


if __name__ == "__main__":

    strProjectFolder = os.path.dirname(__file__)

    DataTrain = open(os.path.join(strProjectFolder, "01-Data/train.csv"), "r")
    DataTest = open(os.path.join(strProjectFolder, "01-Data/test.csv"), "r")

    listTrainLabel, listTrainImageVector, listTrainImage = makeDataProcessing(DataTrain)
    np.savez(os.path.join(strProjectFolder, "01-Data/Train.npz"), Label=np.asarray(listTrainLabel), Image=np.asarray(listTrainImage))

    listTestLabel, listTestImageVector, listTestImage = makeDataProcessing(DataTest)
    np.savez(os.path.join(strProjectFolder, "01-Data/Test.npz"), Label=np.asarray(listTestLabel), Image=np.asarray(listTestImage))
