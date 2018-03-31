import os
import pandas as pd
from Sources import Train, Plot


def main(boolBias, boolNormalize):
    strProjectFolder = os.path.dirname(__file__)

    if boolBias:
        if boolNormalize:
            strOutputPath = "02-Output/" + "Bias" + "Normal"
        else:
            strOutputPath = "02-Output/" + "Bias"
    else:
        if boolNormalize:
            strOutputPath = "02-Output/" + "unBias" + "Normal"
        else:
            strOutputPath = "02-Output/" + "unBias"

    DataTrain = pd.read_csv(os.path.join(strProjectFolder, "01-Data/train.csv"), usecols=["UserID", "MovieID", "Rating"])
    DataTrain = DataTrain.sample(frac=1)
    intUserSize = len(DataTrain["UserID"].drop_duplicates())
    intMovieSize = len(DataTrain["MovieID"].drop_duplicates())

    Users = DataTrain["UserID"].values
    Movies = DataTrain["MovieID"].values
    Rate = DataTrain["Rating"].values

    # Train.getTrain(arrayUser=Users, arrayMovie=Movies, arrayRate=Rate, intUserSize=intUserSize, intMovieSize=intMovieSize, boolBias=boolBias, boolNormalize=boolNormalize, strProjectFolder=strProjectFolder, strOutputPath=strOutputPath)

    Plot.plotModel(strProjectFolder, strOutputPath)
    # Plot.plotLossAccuracyCurves(strProjectFolder, strOutputPath)

if __name__ == "__main__":
    main(boolBias=True, boolNormalize=True)



