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


dfTrainX = pd.read_csv(os.path.join(os.path.dirname(__file__), "X_train_my.csv"))
dfTrainY = pd.read_csv(os.path.join(os.path.dirname(__file__), "Y_train_my.csv"))
dfTestX = pd.read_csv(os.path.join(os.path.dirname(__file__), "X_test_my.csv"))

arrayTrainX = np.array(dfTrainX.values) # (32561, 106)
arrayTestX = np.array(dfTestX.values) # (16281, 106)
arrayTrainY = np.array(dfTrainY.values) # (32561)

arrayNormalizeTrainX, arrayNormalizeTestX = getNormalizeData(arrayTrainX, arrayTestX)

arrayTrainAllX, arrayTrainAllY = getShuffleData(arrayNormalizeTrainX, arrayTrainY)

arrayTrainX, arrayTrainY, arrayValidX, arrayValidY = getTrainAndValidData(arrayTrainAllX, arrayTrainAllY, 0.5)


###---Train(mini batch gradient descent)---###
intEpochNum = 1000
intBatchSize = 32
floatLearnRate = 0.1

arrayW = np.zeros((arrayTrainX.shape[1])) # (106, )
arrayB = np.zeros(1) # (1, )

floatTotalLoss = 0.0
for epoch in range(1, intEpochNum):

    if epoch % 50 == 0:
        print("Epoch:{}, Epoch average loss:{} ".format(epoch, float(floatTotalLoss) / (50.0*len(arrayTrainX))))
        floatTotalLoss = 0.0
        z = np.dot(arrayValidX, arrayW) + arrayB
        result = ((np.around(getSigmoidValue(z))) == np.squeeze(arrayValidY))
        print(float(result.sum())/ len(arrayValidY))

    arrayTrainX, arrayTrainY = getShuffleData(arrayX=arrayTrainX, arrayY=arrayTrainY)

    for batch in range(int(len(arrayTrainX)/intBatchSize)):
        X = arrayTrainX[intBatchSize*batch:intBatchSize*(batch+1)] # (intBatchSize, 106)
        Y = arrayTrainY[intBatchSize*batch:intBatchSize*(batch+1)] # (intBatchSize, 1)

        z = np.dot(X, arrayW) + arrayB
        s = getSigmoidValue(z)

        arrayCrossEntropy = -1 * (np.dot(np.transpose(Y), np.log(s)) + np.dot((1-np.transpose(Y)), np.log(1-s)))

        floatTotalLoss += arrayCrossEntropy

        arrayGradientW = np.mean(-1 * X * (np.squeeze(Y) - s).reshape((intBatchSize,1)), axis=0) # need check
        # arrayGradientW = -1 * np.dot(np.transpose(X), (np.squeeze(Y) - s).reshape((intBatchSize,1))) 
        arrayGradientB = np.mean(-1 * (np.squeeze(Y) - s))
    
        arrayW -= floatLearnRate * np.squeeze(arrayGradientW)
        arrayB -= floatLearnRate * arrayGradientB

    # print("Epoch:{}, CrossEntropy:{} , TotalLoss{} ".format(epoch, arrayCrossEntropy, floatTotalLoss))


###---Test---###
ans = pd.read_csv(os.path.join(os.path.dirname(__file__), "correct_answer.csv"))
Testz = (np.dot(arrayNormalizeTestX, arrayW) + arrayB)
predict = np.around(getSigmoidValue(Testz))


dictD = {"Predict":predict, "Target":ans["label"]}
ResultTable = pd.DataFrame(dictD, columns=dictD.keys())
print(ResultTable)

print(ans["label"].value_counts())
result = ((predict) == np.squeeze(ans["label"]))
print(sum(result)/ len(ans))

