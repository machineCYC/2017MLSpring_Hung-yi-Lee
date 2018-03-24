import os
from Model import MF
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger


def getTrain(arrayUser, arrayMovie, arrayRate, intUserSize, intMovieSize, strProjectFolder, strOutputPath):

    intLatentSize = 32

    model = MF(intUserSize=intUserSize, intMovieSize=intMovieSize, intLatentSize=intLatentSize)

    callbacks = [EarlyStopping("val_loss", patience=50), ModelCheckpoint(os.path.join(strProjectFolder, strOutputPath + "model.h5"), save_best_only=True), CSVLogger(os.path.join(strProjectFolder, strOutputPath + "log.csv"), separator=",", append=False)]
    
    model.fit([arrayUser, arrayMovie], arrayRate, epochs=1000, batch_size=10000, verbose=2, validation_split=.1, callbacks=callbacks)

    model.save(os.path.join(strProjectFolder, strOutputPath + "model.h5"))
    