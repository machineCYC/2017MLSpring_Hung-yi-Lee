import csv, os
import pandas as pd
import matplotlib.pyplot as plt   
from keras.utils import plot_model
from keras.models import load_model


def plotModel(mode):
    strProjectFolder = os.path.dirname(__file__)

    emotion_classifier = load_model(os.path.join(strProjectFolder, "02-Output/"+mode+"model.h5"))
    emotion_classifier.summary()
    plot_model(emotion_classifier, show_shapes=True, to_file=os.path.join(strProjectFolder, "02-Output/"+mode+"model.png"))


def plotLossAndAccuracyCurves(mode):
    strProjectFolder = os.path.dirname(__file__)

    pdLog = pd.read_csv(os.path.join(strProjectFolder, "02-Output/"+mode+"log.csv"))

    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(pdLog["epoch"], pdLog["loss"], "r", linewidth=3.0)
    plt.plot(pdLog["epoch"], pdLog["val_loss"], "b", linewidth=3.0)
    plt.legend(["Training loss", "Validation Loss"], fontsize=18)
    plt.xlabel("Epochs ", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.title("Loss Curves", fontsize=16)
    plt.savefig(os.path.join(strProjectFolder, "02-Output/"+mode+"LossCurves"))

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(pdLog["epoch"], pdLog["acc"], "r", linewidth=3.0)
    plt.plot(pdLog["epoch"], pdLog["val_acc"], "b", linewidth=3.0)
    plt.legend(["Training Accuracy", "Validation Accuracy"], fontsize=18)
    plt.xlabel("Epochs ", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.title("Accuracy Curves", fontsize=16)
    plt.savefig(os.path.join(strProjectFolder, "02-Output/"+mode+"AccuracyCurves"))
    plt.show()

    def plotConfusionMatrix():
        pass