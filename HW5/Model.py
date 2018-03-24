from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Flatten, Dot, Add, Merge, Concatenate, Reshape
from keras.optimizers import Adam
from Base import getRMSE


def MF(intUserSize, intMovieSize, intLatentSize):
    UserInput = Input(shape=(1, ))
    UserEmbedding = Embedding(intUserSize, intLatentSize, embeddings_initializer="random_normal")(UserInput)
    UserEmbedding = Flatten()(UserEmbedding)
    UserBias = Embedding(intUserSize, 1, embeddings_initializer="zeros")(UserInput)
    UserBias = Flatten()(UserBias)

    MovieInput = Input(shape=(1, ))
    MovieEmbedding = Embedding(intMovieSize, intLatentSize, embeddings_initializer="random_normal")(MovieInput)
    MovieEmbedding = Flatten()(MovieEmbedding)
    MovieBias = Embedding(intMovieSize, 1, embeddings_initializer="zeros")(MovieInput)
    MovieBias = Flatten()(MovieBias)

    RattingHat = Dot(axes=1)([UserEmbedding, MovieEmbedding])
    RattingHat = Add()([RattingHat, UserBias, MovieBias])

    model = Model([UserInput, MovieInput], RattingHat)

    optim = Adam()

    model.compile(optimizer=optim, loss="mean_squared_error")
    model.summary()
    return model
    
