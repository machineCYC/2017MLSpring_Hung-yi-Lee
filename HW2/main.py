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









