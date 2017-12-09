import csv, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def getShuffleData(arrayX, arrayY):
    arrayRandomIndex = np.arange(len(arrayX))
    np.random.shuffle(arrayRandomIndex)
    return arrayX[arrayRandomIndex], arrayY[arrayRandomIndex]


def getNormalizeData(arrayTrainX, arrayTestX):
    arrayX = np.concatenate((arrayTrainX, arrayTestX))
    
    arrayMuX = np.mean(arrayX, axis=0)
    arraySigmaX = np.std(arrayX, axis=0)

    arrayNormalizeX = (arrayX - arrayMuX) / arraySigmaX

    arrayNormalizeTrainX, arrayNormalizeTestX = arrayNormalizeX[0:arrayTrainX.shape[0]], arrayNormalizeX[arrayTrainX.shape[0]:]
    return arrayNormalizeTrainX, arrayNormalizeTestX


def getSigmoidValue(z):
    s = 1 / (1.0 + np.exp(-z))
    return np.clip(s, 1e-8, 1 - (1e-8))


def getTrainAndValidData(arrayTrainAllX, arrayTrainAllY, percentage):
    intInputDataSize = len(arrayTrainAllX)
    intValidDataSize = int(np.floor(intInputDataSize * percentage))

    arrayTrainAllX, arrayTrainAllY = getShuffleData(arrayTrainAllX, arrayTrainAllY)

    arrayValidX = arrayTrainAllX[0:intValidDataSize]
    arrayTrainX = arrayTrainAllX[intValidDataSize:]

    arrayValidY = arrayTrainAllY[0:intValidDataSize]
    arrayTrainY = arrayTrainAllY[intValidDataSize:]
    return arrayTrainX, arrayTrainY, arrayValidX, arrayValidY


if __name__ == "__main__":
    
    # read Training data, Training label, Testing data
    dfTrainX = pd.read_csv(os.path.join(os.path.dirname(__file__), "X_train_my.csv"))
    dfTrainY = pd.read_csv(os.path.join(os.path.dirname(__file__), "Y_train_my.csv"))
    dfTestX = pd.read_csv(os.path.join(os.path.dirname(__file__), "X_test_my.csv"))

    # transform the data to array
    arrayTrainX = np.array(dfTrainX.values) # (32561, 106)
    arrayTestX = np.array(dfTestX.values) # (16281, 106)
    arrayTrainY = np.array(dfTrainY.values) # (32561)

    # normalize the Training and Testing data
    arrayNormalizeTrainX, arrayNormalizeTestX = getNormalizeData(arrayTrainX, arrayTestX)

    # Shuffling data index
    arrayTrainAllX, arrayTrainAllY = getShuffleData(arrayNormalizeTrainX, arrayTrainY)

    # take some training data to be validation data
    arrayTrainX, arrayTrainY, arrayValidX, arrayValidY = getTrainAndValidData(arrayTrainAllX, arrayTrainAllY, 0.2)


    ###---Train(mini batch gradient descent)---###
    intBatchIterationNum = 100
    intEpochNum = int(len(arrayTrainX)/intBatchIterationNum)
    intBatchSize = 32
    floatLearnRate = 0.1

    arrayW = np.zeros((arrayTrainX.shape[1])) # (106, )
    arrayB = np.zeros(1) # (1, )

    floatTotalLoss = 0.0
    for epoch in range(1, intEpochNum):

        if epoch % 10 == 0:
            print("Epoch:{}, Epoch average loss:{} ".format(epoch, float(floatTotalLoss) / (10.0*len(arrayTrainX))))
            floatTotalLoss = 0.0
            z = np.dot(arrayValidX, arrayW) + arrayB
            result = ((np.around(getSigmoidValue(z))) == np.squeeze(arrayValidY))
            print("Accuracy:{} ".format(float(result.sum())/ len(arrayValidY)))

        arrayTrainX, arrayTrainY = getShuffleData(arrayX=arrayTrainX, arrayY=arrayTrainY)

        for batch_iter in range(intBatchIterationNum):
            X = arrayTrainX[intBatchSize*batch_iter:intBatchSize*(batch_iter+1)] # (intBatchSize, 106)
            Y = arrayTrainY[intBatchSize*batch_iter:intBatchSize*(batch_iter+1)] # (intBatchSize, 1)

            z = np.dot(X, arrayW) + arrayB
            s = getSigmoidValue(z)

            arrayCrossEntropy = -1 * (np.dot(np.transpose(Y), np.log(s)) + np.dot((1-np.transpose(Y)), np.log(1-s)))

            floatTotalLoss += arrayCrossEntropy

            # arrayGradientW = np.mean(-1 * X * (np.squeeze(Y) - s).reshape((intBatchSize,1)), axis=0) # need check
            arrayGradientW = -1 * np.dot(np.transpose(X), (np.squeeze(Y) - s).reshape((intBatchSize,1))) 
            arrayGradientB = np.mean(-1 * (np.squeeze(Y) - s))
        
            arrayW -= floatLearnRate * np.squeeze(arrayGradientW)
            arrayB -= floatLearnRate * arrayGradientB

        print("CrossEntropy:{} , TotalLoss{} ".format(arrayCrossEntropy, floatTotalLoss))


    ###---Test---###
    ans = pd.read_csv(os.path.join(os.path.dirname(__file__), "correct_answer.csv"))
    Testz = (np.dot(arrayNormalizeTestX, arrayW) + arrayB)
    predict = np.around(getSigmoidValue(Testz))


    dictD = {"Predict":predict, "Target":ans["label"]}
    ResultTable = pd.DataFrame(dictD, columns=dictD.keys())
    print(ResultTable)

    result = ((predict) == np.squeeze(ans["label"]))
    print("Testing Accuracy:{} ".format(sum(result)/ len(ans)))

