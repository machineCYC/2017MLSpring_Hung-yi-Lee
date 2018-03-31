from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Flatten, Dot, Add, Dropout
from keras.optimizers import Adam
from Sources import Base


def MF(intUserSize, intMovieSize, intLatentSize, boolBias):
    UserInput = Input(shape=(1, ), name="UserInput")
    UserEmbedding = Embedding(intUserSize, intLatentSize, embeddings_initializer="random_normal", name="UserEmbedding")(UserInput)
    UserEmbedding = Flatten()(UserEmbedding)
    UserEmbedding = Dropout(0.8)(UserEmbedding)

    MovieInput = Input(shape=(1, ), name="MovieInput")
    MovieEmbedding = Embedding(intMovieSize, intLatentSize, embeddings_initializer="random_normal", name="MovieEmbedding")(MovieInput)
    MovieEmbedding = Flatten()(MovieEmbedding)
    MovieEmbedding = Dropout(0.8)(MovieEmbedding)

    RattingHat = Dot(axes=1)([UserEmbedding, MovieEmbedding], name="Ratting")

    if boolBias:
        UserBias = Embedding(intUserSize, 1, embeddings_initializer="zeros", name="UserBias")(UserInput)
        UserBias = Flatten()(UserBias)

        MovieBias = Embedding(intMovieSize, 1, embeddings_initializer="zeros", name="MovieBias")(MovieInput)
        MovieBias = Flatten()(MovieBias)

        RattingHat = Add()([RattingHat, UserBias, MovieBias])

    model = Model([UserInput, MovieInput], RattingHat)

    optim = Adam()
    model.compile(optimizer=optim, loss="mse", metrics=[Base.getRMSE])
    model.summary()
    return model
    
