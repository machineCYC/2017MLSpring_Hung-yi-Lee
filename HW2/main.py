import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


###---Data Processing---###
dfData = pd.read_csv("D:/Git/2017MLSpring_Hung-yi-Lee/HW2/train.csv")
dfDataX = dfData.drop(["income", "education_num", "sex"], axis=1)
dfDataY = dfData["income"]

listObjectColumnName = [col for col in dfDataX.columns if dfDataX[col].dtypes=="object"]
listNonObjectColumnName = [col for col in dfDataX.columns if dfDataX[col].dtypes!="object"]

dfNonObjectData = dfDataX[listNonObjectColumnName]
dfNonObjectData.insert(2, "sex", (dfData["sex"]==" Male").astype(np.int)) # Male 1 Femal 0

dfObjectData = dfDataX[listObjectColumnName]
dfObjectData = pd.get_dummies(dfObjectData)

dfTrainX = dfNonObjectData.join(dfObjectData)
dfTrainX = dfTrainX.astype("int64")
dfTrainY = (dfDataY==" >50K").astype("int64") # >50K 1, =<50K 0

arrayTrainX = np.array(dfTrainX.values) # (32561, 106)
arrayTrainY = np.array(dfTrainY.values) # (32561, )


def getShuffleData(arrayX, arrayY):
    arrayRandomIndex = np.arange(len(arrayX))
    np.random.shuffle(arrayRandomIndex)

    arrayX = arrayX[arrayRandomIndex]
    arrayY = arrayY[arrayRandomIndex]

    return arrayX, arrayY

# arrayRandomIndex = np.arange(len(arrayTrainX))
# np.random.shuffle(arrayRandomIndex)

# arrayTrainX = arrayTrainX[arrayRandomIndex]
# arrayTrainY = arrayTrainY[arrayRandomIndex]

def getNormalizeData(arrayX):
    arrayMuX = np.mean(arrayX, axis=0)
    arraySigmaX = np.std(arrayX, axis=0)

    arrayNormalizeX = (arrayX - arrayMuX) / arraySigmaX
    return arrayNormalizeX

# arrayTrainMuX = np.mean(arrayTrainX, axis=0)
# arrayTrainSigmaX = np.std(arrayTrainX, axis=0)
# arrayTrainNormalizeX = (arrayTrainX - arrayTrainMuX) / arrayTrainSigmaX

def getSigmoidValue(z):
    s = 1 / (1.0 + np.exp(-z))
    return np.clip(s, 1e-8, 1 - (1e-8))

intEpochNum = 1000
intBatchSize = 32
floatTotalLoss = 0.0
floatLearnRate = 0.1

arrayW = np.zeros(arrayTrainX.shape[1]) # (106, )
arrayB = np.zeros(1) # (1, )

    
arrayTrainAllX, arrayTrainAllY = getShuffleData(arrayX=arrayTrainX, arrayY=arrayTrainY)

arrayTrainAllNormalX = getNormalizeData(arrayX=arrayTrainAllX)

intTrainALLDataSize = len(arrayTrainAllNormalX)
intValidDataSize = int(np.floor(intTrainALLDataSize * 0.3))

for epoch in range(1, intEpochNum):

    arrayValidX = arrayTrainAllNormalX[0:intValidDataSize]
    arrayTrainX = arrayTrainAllNormalX[intValidDataSize:]

    arrayValidY = arrayTrainAllY[0:intValidDataSize]
    arrayTrainY = arrayTrainAllY[intValidDataSize:]

    for batch in range(int(intTrainALLDataSize/intBatchSize)):
        X = arrayTrainX[intBatchSize*batch:intBatchSize*(batch+1)] # (intBatchSize, 163)
        Y = arrayTrainY[intBatchSize*batch:intBatchSize*(batch+1)] # (intBatchSize,)

        z = X.dot(arrayW) + arrayB
        s = getSigmoidValue(z)

        arrayCrossEntropy = -1 * (Y.dot(np.log(s)) + (1-Y).dot(np.log(1-s)))

        floatTotalLoss += arrayCrossEntropy

        arrayGradientW = X.T.dot(Y - s)
        arrayGradientB = np.mean(Y - s)

        arrayW -= floatLearnRate * arrayGradientW
        arrayB -= floatLearnRate * arrayGradientB

    print("Epoch:{}, CrossEntropy:{} ".format(epoch, arrayCrossEntropy))

print("GG")






# TrainDataSize = len(arrayTrainNormalizeX)
# ValidDataSize = int(floor(TrainDataSize * 0.3))

# arrayValidX = arrayTrainNormalizeX[0:ValidDataSize, :]
# arrayTrainNormalizeX2 = arrayTrainNormalizeX[ValidDataSize:, :]

# arrayValidY = arrayTrainY[0:ValidDataSize, :]
# arrayTrainY2 = arrayTrainY[ValidDataSize:, :]








# dfTrainX.to_csv("D:/Git/2017MLSpring_Hung-yi-Lee/HW2/X_train_my")



# print(dfTrainX)
# data = pd.read_csv("D:/Git/2017MLSpring_Hung-yi-Lee/HW2/X_train")
# for c in data.columns:
#     print(c)

# print(data["?_occupation"])

# ?_native_country
# ?_workclass



