import os, keras
import numpy as np
from keras.callbacks import CSVLogger
from Model import buildModel
from Plot import PlotLossAndAccuracyCurves


def main(mode):

    intVaildSize = 5000
    intEpochs = 100
    intBatchSize = 64
    intImageWide = 48
    intImageLength = 48

    strProjectFolder = os.path.dirname(__file__)

    DataTrain = np.load(os.path.join(strProjectFolder, "01-Data/Train.npz"))

    arrayLabel = keras.utils.to_categorical(DataTrain["Label.npy"])

    if mode == "cnn":
        arrayImage = DataTrain["Image.npy"]/255.
    else:
        arrayImage = DataTrain["Image.npy"].reshape(-1, 48*48)/255.

    arrayTrainX, arrayTrainY = arrayImage[:-intVaildSize], arrayLabel[:-intVaildSize]
    arrayValidX, arrayValidY = arrayImage[-intVaildSize:], arrayLabel[-intVaildSize:]

    model = buildModel(mode=mode)
    # keras.utils.plot_model(model, to_file=os.path.join(strProjectFolder, "02-Output/model.png"), show_shapes=True)

    callbacks = []
    csvLogger = CSVLogger(os.path.join(strProjectFolder, "02-Output/"+mode+"log.csv"), separator=",", append=False)
    callbacks.append(csvLogger)

    model.fit(arrayTrainX, arrayTrainY, epochs=intEpochs, batch_size=intBatchSize, verbose=2, validation_data=(arrayValidX, arrayValidY), callbacks=callbacks, shuffle=True)

    
    model.save(os.path.join(strProjectFolder, "02-Output/"+mode+"model.h5"))
    
    PlotLossAndAccuracyCurves(mode)


if __name__ == "__main__":
    main(mode="dnn")


