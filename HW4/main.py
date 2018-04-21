import os
from Base import DataProcessing, Train


strProjectFolder = os.path.dirname(__file__)
strOutputFolder = os.path.join(strProjectFolder, "03-Output")

intSequenceLength = 40
intVocabSize = 20000
intEmbeddingDim = 128
intHiddenSize = 64
floatRatio = 0.1


ETL = DataProcessing.executeETL()
ETL.cleanData(strDataFile="training_label.txt", boolLabel=True)
ETL.doTokenizer(intVocabSize=intVocabSize)
# ETL.saveTokenizer(strTokenizerFile="TokenizerDictionary")
ETL.loadTokenizer(strTokenizerFile="TokenizerDictionary")
ETL.convertWords2Sequence(intSequenceLength=intSequenceLength)

arrayTrain, arrayValid = ETL.splitData(floatRatio=floatRatio)

Train.getTrain(intSequenceLength=intSequenceLength, intVocabSize=intVocabSize, intEmbeddingDim=intEmbeddingDim, intHiddenSize=intHiddenSize, arrayTrain=arrayTrain, arrayValid=arrayValid)





