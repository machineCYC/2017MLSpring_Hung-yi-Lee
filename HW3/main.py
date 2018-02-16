import os, keras
import numpy as np
from keras.callbacks import CSVLogger
from Model import buildModel
from Plot import plotLossAndAccuracyCurves, plotModel, plotConfusionMatrix, plotImageFiltersResult
from Predict import makePredict

from sklearn.metrics import confusion_matrix

# from keras.models import load_model
# from keras import backend as K
# import matplotlib.pyplot as plt 

def main(mode):

    intVaildSize = 5000
    intEpochs = 100
    intBatchSize = 128

    strProjectFolder = os.path.dirname(__file__)

    DataTrain = np.load(os.path.join(strProjectFolder, "01-Data/Train.npz"))

    arrayLabel = DataTrain["Label.npy"]
    arrayOneHotLabel = keras.utils.to_categorical(arrayLabel)
    arrayImage = DataTrain["Image.npy"]/255.

    arrayTrainX, arrayTrainY, arrayTrainLabel = arrayImage[:-intVaildSize], arrayOneHotLabel[:-intVaildSize], arrayLabel[:-intVaildSize]
    arrayValidX, arrayValidY, arrayValidLabel = arrayImage[-intVaildSize:], arrayOneHotLabel[-intVaildSize:], arrayLabel[-intVaildSize:]

    model = buildModel(mode=mode)
    # keras.utils.plot_model(model, to_file=os.path.join(strProjectFolder, "02-Output/model.png"), show_shapes=True)

    callbacks = []
    csvLogger = CSVLogger(os.path.join(strProjectFolder, "02-Output/"+mode+"log.csv"), separator=",", append=False)
    callbacks.append(csvLogger)

    model.fit(arrayTrainX, arrayTrainY, epochs=intEpochs, batch_size=intBatchSize, verbose=2, validation_data=(arrayValidX, arrayValidY), callbacks=callbacks, shuffle=True)

    model.save(os.path.join(strProjectFolder, "02-Output/"+mode+"model.h5"))
    
    plotLossAndAccuracyCurves(mode)
    plotModel(mode)
 
    arrayPred = makePredict(mode, arrayX=arrayValidX)
    arrayPredLabel = np.argmax(arrayPred, axis=1)
    arrayConfusionMatrix = confusion_matrix(arrayValidLabel, arrayPredLabel)
    plotConfusionMatrix(mode, arrayConfusionMatrix, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])

    if mode=="cnn":
        plotImageFiltersResult(mode, arrayX=arrayValidX, intChooseId=0)



if __name__ == "__main__":
    main(mode="cnn")


