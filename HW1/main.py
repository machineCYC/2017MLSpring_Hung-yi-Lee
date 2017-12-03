import numpy as np
import csv
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pandas as pd

###---DataProcessing---###
# 給定資料空間
listTrainData = []
for i in range(18):
	listTrainData.append([])

# 將資料放進空間
textTrain = open("D:/Git/2017MLSpring_Hung-yi-Lee/HW1/train.csv", "r", encoding="big5") 
rowTrain = csv.reader(textTrain)
n_row = 0
for r in rowTrain:
    if n_row != 0:
        for i in range(3, 27):
            if r[i] != "NR":
                listTrainData[(n_row-1) % 18].append(float(r[i]))
            else:
                listTrainData[(n_row-1) % 18].append(float(0))   
    n_row += 1    
textTrain.close()

listTrainX = []
listTrainY = []
# 將資料拆成 x 和 y
for m in range(12):
    # 一個月每10小時算一筆資料，會有471筆
    for i in range(471):
        listTrainX.append([])
        listTrainY.append(listTrainData[9][480*m + i + 9])
        # 18種汙染物
        for p in range(18):
        # 收集9小時的資料
            for t in range(9):
                listTrainX[471*m + i].append(listTrainData[p][480*m + i + t])

###---Train---###
arrayTrainX = np.array(listTrainX)
arrayTrainY = np.array(listTrainY)
# 增加bias項
arrayTrainX = np.concatenate((np.ones((arrayTrainX.shape[0], 1)), arrayTrainX), axis=1) # (5652, 163)

# Adagrad
intLearningRate = 5
Iteration = 20000

arrayW = np.zeros(arrayTrainX.shape[1])  # (163, )
arrayGradientSum = np.zeros(arrayTrainX.shape[1])
listCost = []
for itera in range(Iteration):
    arrayYHat = arrayTrainX.dot(arrayW)
    arrayLoss = arrayYHat - arrayTrainY
    arrayCost = np.sum(arrayLoss**2) / arrayTrainX.shape[0]

    # save cost function value in process
    listCost.append(arrayCost)

    arrayGradient = arrayTrainX.T.dot(arrayLoss) / arrayTrainX.shape[0]
    arrayGradientSum += arrayGradient**2
    arraySigma = np.sqrt(arrayGradientSum)
    arrayW -= intLearningRate * arrayGradient / arraySigma

    if itera % 1000 == 0:
        print("iteration:{}, cost:{} ".format(itera, arrayCost))

###---Test---###
listTestData = []
textTest = open("D:/Git/2017MLSpring_Hung-yi-Lee/HW1/test.csv", "r", encoding="big5")
rowTest = csv.reader(textTest)
n_row = 0
for r in rowTest:
    if n_row % 18 == 0:
        listTestData.append([])
        for i in range(2, 11):
            listTestData[n_row // 18].append(float(r[i]))
    else:
        for i in range(2, 11):
            if r[i] == "NR":
                listTestData[n_row // 18].append(float(0))
            else:
                listTestData[n_row // 18].append(float(r[i]))
    n_row += 1
textTest.close()

arrayTestX = np.array(listTestData)
arrayTestX = np.concatenate((np.ones((arrayTestX.shape[0], 1)), arrayTestX), axis=1)  # (240, 163)
arrayPredictY = np.dot(arrayTestX, arrayW)

# close form
arrayCloseFormW = inv(arrayTrainX.T.dot(arrayTrainX)).dot(arrayTrainX.T.dot(arrayTrainY))
arrayPredictCloseY = np.dot(arrayTestX, arrayCloseFormW)

###---Visualization---###
plt.plot(np.arange(len(listCost[100:])),listCost[100:], "b--")
plt.title("Train Process")
plt.xlabel("Adagrad Iteration")
plt.ylabel("Cost Function (MSE)")
plt.show()

dcitD = {"Adagrad":arrayPredictY, "CloseForm":arrayPredictCloseY}
pdResult = pd.DataFrame(dcitD)
print(pdResult)

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(np.arange(len(arrayPredictY)), arrayPredictY, "b--")
plt.title("Adagrad")
plt.xlabel("Test Data Index")
plt.ylabel("Predict Result")
plt.subplot(122)
plt.plot(np.arange(len(arrayPredictCloseY)), arrayPredictCloseY, "r--")
plt.title("CloseForm")
plt.xlabel("Test Data Index")
plt.ylabel("Predict Result")
plt.show()

#%%
# gradient decent
"""
使用gradient decent learning rate 要調很小，不然很容易爆炸
"""
intLearningRate = 1e-8
Iteration = 100000

listCost = []
for itera in range(Iteration):
    arrayYHat = x.dot(arrayW)
    arrayLoss = arrayYHat - y
    arrayCost = np.sum(arrayLoss**2) / x.shape[0]
    listCost.append(arrayCost)

    arrayGradient = x.T.dot(arrayLoss) / x.shape[0]
    arrayW -= intLearningRate * arrayGradient
    if itera % 10000 == 0:
        print("iteration:{}, cost:{} ".format(itera, arrayCost))
plt.plot(np.arange(len(listCost)),listCost, "b--")
plt.show()
