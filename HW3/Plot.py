import csv, os, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   
from keras.utils import plot_model
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from keras import backend as K


def plotDigits(instances, images_per_row=5, size=140,  **options):
    """
    This function prints and plots the lots of images together.
    """
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


def plotModel(mode):
    """
    This function prints and plots the model structure.
    """
    strProjectFolder = os.path.dirname(__file__)

    emotion_classifier = load_model(os.path.join(strProjectFolder, "02-Output/"+mode+"model.h5"))
    emotion_classifier.summary()
    plot_model(emotion_classifier, show_shapes=True, to_file=os.path.join(strProjectFolder, "02-Output/"+mode+"model.png"))


def plotLossAndAccuracyCurves(mode):
    """
    This function prints and plots the Loss Curves and Accuracy Curves.
    """
    strProjectFolder = os.path.dirname(__file__)

    pdLog = pd.read_csv(os.path.join(strProjectFolder, "02-Output/"+mode+"log.csv"))

    # Loss Curves
    plt.figure(figsize=(8, 6))
    plt.plot(pdLog["epoch"], pdLog["loss"], "r", linewidth=2.0)
    plt.plot(pdLog["epoch"], pdLog["val_loss"], "b", linewidth=2.0)
    plt.legend(["Training loss", "Validation Loss"], fontsize=18)
    plt.xlabel("Epochs ", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.title("Loss Curves", fontsize=16)
    plt.savefig(os.path.join(strProjectFolder, "02-Output/"+mode+"LossCurves"))

    # Accuracy Curves
    plt.figure(figsize=(8, 6))
    plt.plot(pdLog["epoch"], pdLog["acc"], "r", linewidth=2.0)
    plt.plot(pdLog["epoch"], pdLog["val_acc"], "b", linewidth=2.0)
    plt.legend(["Training Accuracy", "Validation Accuracy"], fontsize=18)
    plt.xlabel("Epochs ", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.title("Accuracy Curves", fontsize=16)
    plt.savefig(os.path.join(strProjectFolder, "02-Output/"+mode+"AccuracyCurves"))



def plotConfusionMatrix(mode, confusionmatrix, classes, title="Confusion matrix", cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    strProjectFolder = os.path.dirname(__file__)
    
    confusionmatrix = confusionmatrix.astype("float") / confusionmatrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    plt.imshow(confusionmatrix, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = confusionmatrix.max() / 2.
    for i, j in itertools.product(range(confusionmatrix.shape[0]), range(confusionmatrix.shape[1])):
        plt.text(j, i, "{:.2f}".format(confusionmatrix[i, j]), horizontalalignment="center",
                color="white" if confusionmatrix[i, j] > thresh else "black")
    plt.tight_layout() # 自動調整間距
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(os.path.join(strProjectFolder, "02-Output/"+mode+title))


def plotImageActivateFilters():
    pass


def plotImageFiltersResult(mode, arrayX, intChooseId=0):
    """
    This function plot the output of convolution layer in valid data image.
    """
    intImageHeight = 48
    intImageWidth = 48

    strProjectFolder = os.path.dirname(__file__)
    strModelPath = os.path.join(strProjectFolder, "02-Output/" + mode + "model.h5")

    model = load_model(strModelPath)
    dictLayer = dict([layer.name, layer] for layer in model.layers)
    inputImage = model.input
    listLayerNames = [layer for layer in dictLayer.keys() if "activation" in layer][:4]
    listCollectLayers = [K.function([inputImage, K.learning_phase()], [dictLayer[name].output]) for name in listLayerNames] # ??

    for cnt, fn in enumerate(listCollectLayers):
        arrayPhoto = arrayX[intChooseId].reshape(1, intImageWidth, intImageHeight, 1)
        listLayerImage = fn([arrayPhoto, 0]) # get the output of that layer list (1, 1, 48, 48, 64)
        fig = plt.figure(figsize=(16, 17))
        intFilters = 64
        for i in range(intFilters):
            ax = fig.add_subplot(intFilters/8, 8, i+1)
            ax.imshow(listLayerImage[0][0, :, :, i], cmap="Blues")
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.xlabel("filter {}".format(i))
            plt.tight_layout()
        fig.suptitle("Output of {} (Given image{})".format(listLayerNames[cnt], intChooseId))

        plt.savefig(os.path.join(strProjectFolder, "02-Output/"+mode+"FiltersResultImage"+str(intChooseId)+listLayerNames[cnt]))

