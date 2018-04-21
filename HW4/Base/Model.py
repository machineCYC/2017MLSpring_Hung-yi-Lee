from keras.layers import Input, LSTM, Dense
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.models import Model
from keras.optimizers import Adam

# , kernel_regularizer=regularizers.l2(0.1)
def RNN(intSequenceLength, intVocabSize, intEmbeddingDim, intHiddenSize):

    inputs = Input(shape=(intSequenceLength,))

    VocabEmbedding = Embedding(intVocabSize, intEmbeddingDim)(inputs)

    cell = LSTM(units=intHiddenSize, return_sequences=False)

    output = cell(VocabEmbedding) 

    output = Dense(intHiddenSize//2, activation="relu")(output)

    output = Dense(1, activation="sigmoid")(output)

    model = Model(inputs=inputs, outputs=output)

    optim = Adam()
    model.compile(optimizer=optim, loss="binary_crossentropy", metrics=["accuracy"])
    return model