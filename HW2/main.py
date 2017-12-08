import csv, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def makeDataProcessing(dfData):
    dfDataX = dfData.drop(["education_num", "sex"], axis=1)

    listObjectColumnName = [col for col in dfDataX.columns if dfDataX[col].dtypes=="object"]
    listNonObjectColumnName = [col for col in dfDataX.columns if dfDataX[col].dtypes!="object"]

    dfNonObjectData = dfDataX[listNonObjectColumnName]
    dfNonObjectData.insert(2, "sex", (dfData["sex"]==" Male").astype(np.int)) # Male 1 Femal 0

    dfObjectData = dfDataX[listObjectColumnName]
    dfObjectData = pd.get_dummies(dfObjectData)

    dfDataX = dfNonObjectData.join(dfObjectData)
    dfDataX = dfDataX.astype("int64")
    return dfDataX


def getShuffleData(arrayX, arrayY):
    arrayRandomIndex = np.arange(len(arrayX))
    np.random.shuffle(arrayRandomIndex)
    return arrayX[arrayRandomIndex], arrayY[arrayRandomIndex]


def getNormalizeData(arrayX):
    arrayMuX = np.mean(arrayX, axis=0)
    arraySigmaX = np.std(arrayX, axis=0)

    arrayNormalizeX = (arrayX - arrayMuX) / arraySigmaX
    return arrayNormalizeX


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


###---Data Processing---###
dfDataTrain = pd.read_csv(os.path.join(os.path.dirname(__file__), "train.csv"))
dfDataTest = pd.read_csv(os.path.join(os.path.dirname(__file__), "test.csv"))

intTrainSize = len(dfDataTrain)
intTestSize = len(dfDataTest)

dfDataTrainY = dfDataTrain["income"]
dfTrainY = (dfDataTrainY==" >50K").astype("int64") # >50K 1, =<50K 0
print(dfTrainY.value_counts())
dfDataTrain = dfDataTrain.drop(["income"], axis=1)
dfAllData = pd.concat([dfDataTrain, dfDataTest], axis=0, ignore_index=True)
dfAllData = makeDataProcessing(dfData=dfAllData)

dfTrainX = dfAllData[0:intTrainSize]
dfTestX = dfAllData[intTrainSize:(intTrainSize + intTestSize)]

arrayTrainX = np.array(dfTrainX.values) # (32561, 106)
arrayTestX = np.array(dfTestX.values) # (16281, 106)
arrayTrainY = np.array(dfTrainY.values).reshape(-1, 1) # (32561, 1)

arrayTrainAllNormalX = getNormalizeData(arrayTrainX)

arrayTrainAllX, arrayTrainAllY = getShuffleData(arrayTrainAllNormalX, arrayTrainY)

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
z = np.dot(arrayTestX, arrayW) + arrayB
predict = np.around(getSigmoidValue(z))

dictD = {"Predict":predict, "Target":ans["label"]}
ResultTable = pd.DataFrame(dictD, columns=dictD.keys())
print(ResultTable)

print(ans["label"].value_counts())
result = ((predict) == np.squeeze(ans["label"]))
print(float(result.sum())/ len(ans))

