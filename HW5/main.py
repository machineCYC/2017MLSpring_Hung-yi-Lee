import os
import pandas as pd
from Train import getTrain


strProjectFolder = os.path.dirname(__file__)
strOutputPath = "02-Output/"

DataTrain = pd.read_csv(os.path.join(strProjectFolder, "01-Data/train.csv"), usecols=["UserID", "MovieID", "Rating"])
DataTrain = DataTrain.sample(frac=1)
pdUserSize = len(DataTrain["UserID"].drop_duplicates())
pdMovieSize = len(DataTrain["MovieID"].drop_duplicates())

Users = DataTrain["UserID"].values
Movies = DataTrain["MovieID"].values
Rate = DataTrain["Rating"].values

getTrain(arrayUser=Users, arrayMovie=Movies, arrayRate=Rate, intUserSize=pdUserSize, intMovieSize=pdMovieSize, strProjectFolder=strProjectFolder, strOutputPath=strOutputPath)


