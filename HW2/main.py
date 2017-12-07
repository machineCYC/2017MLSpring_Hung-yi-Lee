import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
b = np.array([[1,1,1,1,1], [2,2,2,2,2]])
a = np.array([2, 3])
c = np.ones(1)
print(b.shape, a.shape)
print(b * a.reshape(2, 1))
# print(np.dot(b, np.transpose(a)) + c)

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
# dfTrainX.to_csv("D:/Git/2017MLSpring_Hung-yi-Lee/HW2/X_train_my",index=False)
# dfTrainY.to_csv("D:/Git/2017MLSpring_Hung-yi-Lee/HW2/Y_train_my",index=False, header=True)

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


# dfTrainX = pd.read_csv("D:/Git/2017MLSpring_Hung-yi-Lee/HW2/X_train_my", sep=',', header=0)
# dfTrainY = pd.read_csv("D:/Git/2017MLSpring_Hung-yi-Lee/HW2/Y_train_my", sep=',', header=0)
# dfTestX = pd.read_csv("D:/Git/2017MLSpring_Hung-yi-Lee/HW2/X_test", sep=',', header=0)

arrayTrainX = np.array(dfTrainX.values) # (32561, 106)
# arrayTestX = np.array(dfTestX.values) # (32561, 106)
arrayTrainY = np.array(dfTrainY.values).reshape(-1, 1) # (32561, 1)


arrayTrainAllNormalX = getNormalizeData(arrayTrainX)

arrayTrainAllX, arrayTrainAllY = getShuffleData(arrayTrainAllNormalX, arrayTrainY)

arrayTrainX, arrayTrainY, arrayValidX, arrayValidY = getTrainAndValidData(arrayTrainAllX, arrayTrainAllY, 0.5)

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

        # arrayGradientW = np.mean(-1 * X * (np.squeeze(Y) - s).reshape((intBatchSize,1)), axis=0) # need check
        arrayGradientW = -1 * np.dot(np.transpose(X), (np.squeeze(Y) - s).reshape((intBatchSize,1))) 
        arrayGradientB = np.mean(-1 * (np.squeeze(Y) - s))
    
        arrayW -= floatLearnRate * np.squeeze(arrayGradientW)
        arrayB -= floatLearnRate * arrayGradientB

    # print("Epoch:{}, CrossEntropy:{} , TotalLoss{} ".format(epoch, arrayCrossEntropy, floatTotalLoss))

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



