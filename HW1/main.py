import numpy as np
import csv
from numpy.linalg import inv
import matplotlib.pyplot as plt

# 給定資料空間
data = []
for i in range(18):
	data.append([])

# 將資料放進空間
text = open("D:/Git/2017MLSpring_Hung-yi-Lee/HW1/train.csv", "r", encoding="big5") 
row = csv.reader(text)
n_row = 0
for r in row:
    if n_row != 0:
        for i in range(3, 27):
            if r[i] != "NR":
                data[(n_row-1) % 18].append(float(r[i]))
            else:
                data[(n_row-1) % 18].append(float(0))   
    n_row += 1    

x = []
y = []
# 將資料拆成 x 和 y
for m in range(12):
    # 一個月每10小時算一筆資料，會有471筆
    for i in range(471):
        x.append([])
        y.append(data[9][480*m + i + 9])
        # 18種汙染物
        for p in range(18):
        # 收集9小時的資料
            for t in range(9):
                x[471*m + i].append(data[p][480*m + i + t])

x = np.array(x)
y = np.array(y)

# 增加bias項
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1) # (5652, 163)

arrayW = np.zeros(x.shape[1]) # (163, )

# Adagrad
intLearningRate = 100
Iteration = 20000
arrayReg = np.zeros(x.shape[1])
listCost = []
for itera in range(Iteration):
    arrayYHat = x.dot(arrayW)
    arrayLoss = arrayYHat - y
    arrayCost = np.sum(arrayLoss**2) / x.shape[0]
    listCost.append(arrayCost)

    arrayGradient = x.T.dot(arrayLoss) / x.shape[0]
    arrayReg += arrayGradient**2
    Ada = np.sqrt(arrayReg)
    arrayW -= intLearningRate * arrayGradient/Ada
    if itera % 1000 == 0:
        print("iteration:{}, cost:{} ".format(itera, arrayCost))
plt.plot(np.arange(len(listCost[100:])),listCost[100:], "b--")
plt.show()

# testing
test_x = []
text = open("D:/Git/2017MLSpring_Hung-yi-Lee/HW1/test.csv", "r", encoding="big5")
row = csv.reader(text)
n_row = 0
for r in row:
    if n_row % 18 ==0:
        text_x.append([])
    for i in range(2, 11):
        text_x[n_row//18].append(float(r[i]))
    n_row += 1
text.close()
test = np.array(text_x)
print(test.shape)
    
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

# close form
# arrayWCloseForm = inv(x.T.dot(x)).dot(x.T.dot(y))
# print(arrayWCloseForm)

#%%
3//4
18%18


