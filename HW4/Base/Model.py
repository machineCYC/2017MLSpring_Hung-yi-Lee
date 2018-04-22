from keras.layers import Input, LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.models import Model
from keras.optimizers import Adam


def RNN(dictModelPara):

    inputs = Input(shape=(dictModelPara["intSequenceLength"],))

    VocabEmbedding = Embedding(dictModelPara["intVocabSize"], dictModelPara["intEmbeddingDim"])(inputs)

    cell = LSTM(units=dictModelPara["intHiddenSize"], return_sequences=False)

    output = cell(VocabEmbedding) 

    output = Dense(dictModelPara["intHiddenSize"]//2, kernel_regularizer=regularizers.l2(0.01), activation="relu")(output)
    output = Dropout(dictModelPara["floatDropoutRate"])(output)
    output = Dense(1, activation="sigmoid")(output)

    model = Model(inputs=inputs, outputs=output)

    optim = Adam()
    model.compile(optimizer=optim, loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model