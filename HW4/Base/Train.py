import os
from Base import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger


strProjectFolder = os.path.dirname(os.path.dirname(__file__))
strOutputFolder = os.path.join(strProjectFolder, "03-Output")


def getTrain(intSequenceLength, intVocabSize, intEmbeddingDim, intHiddenSize, arrayTrain, arrayValid):

    model = Model.RNN(intSequenceLength=intSequenceLength, intVocabSize=intVocabSize, intEmbeddingDim=intEmbeddingDim, intHiddenSize=intHiddenSize)

    callbacks = [EarlyStopping("val_loss", patience=20)
               , ModelCheckpoint(os.path.join(strOutputFolder, "model.h5"), save_best_only=True)
               , CSVLogger(os.path.join(strOutputFolder, "log.csv"), separator=",", append=False)]

    model.fit(x=arrayTrain[0], y=arrayTrain[1], epochs=20, batch_size=128, verbose=2, validation_data=arrayValid, callbacks=callbacks)

    model.save(os.path.join(strOutputFolder, "model.h5"))