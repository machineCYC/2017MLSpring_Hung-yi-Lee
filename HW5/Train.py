import os
import numpy as np
from Sources import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger


def getTrain(arrayUser, arrayMovie, arrayRate, intUserSize, intMovieSize, boolBias, boolNormalize, strProjectFolder, strOutputPath):

    intLatentSize = 32
    intVaildSize = 80000
    arrayTrainUser = arrayUser[:-intVaildSize]
    arrayTrainMovie = arrayMovie[:-intVaildSize]
    arrayTrainRate = arrayRate[:-intVaildSize]

    arrayValidUser = arrayUser[-intVaildSize:]
    arrayValidMovie = arrayMovie[-intVaildSize:]
    arrayValidRate = arrayRate[-intVaildSize:]

    arrayTrainRateAvg = np.mean(arrayTrainRate)
    arrayTrainRateStd = np.std(arrayTrainRate)

    if boolNormalize:
        arrayTrainRate = (arrayTrainRate - arrayTrainRateAvg)/arrayTrainRateStd
        arrayValidRate = (arrayValidRate - arrayTrainRateAvg)/arrayTrainRateStd

    model = Model.MF(intUserSize=intUserSize, intMovieSize=intMovieSize, intLatentSize=intLatentSize, boolBias=boolBias)

    callbacks = [EarlyStopping("val_loss", patience=25), ModelCheckpoint(os.path.join(strProjectFolder, strOutputPath + "model.h5"), save_best_only=True), CSVLogger(os.path.join(strProjectFolder, strOutputPath + "log.csv"), separator=",", append=False)]
    
    model.fit([arrayTrainUser, arrayTrainMovie], arrayTrainRate, epochs=100, batch_size=4096, verbose=2, validation_data=([arrayValidUser, arrayValidMovie], arrayValidRate), callbacks=callbacks)

    model.save(os.path.join(strProjectFolder, strOutputPath + "model.h5"))
    return arrayTrainRateAvg, arrayTrainRateStd
    