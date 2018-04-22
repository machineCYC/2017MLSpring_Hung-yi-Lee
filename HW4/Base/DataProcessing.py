import os, json
from keras.preprocessing.text import Tokenizer
import _pickle as pk
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import to_categorical


strProjectFolder = os.path.dirname(os.path.dirname(__file__))
strRAWDataFolder = os.path.join(strProjectFolder, "01-RAWData")
strAPDataFolder = os.path.join(strProjectFolder, "02-APData")

class executeETL():
    def __init__(self):
        self.dictTrainData = {}

    def cleanData(self, strDataFileName, boolLabel):
        listLabel = []
        listText = []
        with open(os.path.join(strRAWDataFolder, strDataFileName), "r", encoding="utf8") as data:
            for d in data:
                if boolLabel:
                    listRow = d.strip().split(" +++$+++ ")
                    listLabel.append(int(listRow[0]))
                    listText.append(listRow[1])
                else:
                    listText.append(d)

            if boolLabel:
                self.dictTrainData["Data"] = [listText, listLabel]
            else:
                self.dictTrainData["Data"] = [listText]

    def doTokenizer(self, intVocabSize):
        self.tokenizer = Tokenizer(num_words=intVocabSize)
        for key in self.dictTrainData:
            listTexts = self.dictTrainData[key][0]
            self.tokenizer.fit_on_texts(listTexts)

    def saveTokenizer(self, strTokenizerFile):
        pk.dump(self.tokenizer, open(os.path.join(strAPDataFolder, strTokenizerFile), "wb"))

    def loadTokenizer(self, strTokenizerFile):
        self.tokenizer = pk.load(open(os.path.join(strAPDataFolder, strTokenizerFile), "rb"))

    def convertWords2Sequence(self, intSequenceLength):
        for key in self.dictTrainData:
            listSequence = self.tokenizer.texts_to_sequences(self.dictTrainData[key][0])
            self.dictTrainData[key][0] = np.array(pad_sequences(listSequence, maxlen=intSequenceLength))
    
    def convertLabel2Onehot(self):
        for key in self.dictTrainData:
            if len(self.dictTrainData[key]) == 2:
                self.dictTrainData[key][1] = np.array(to_categorical(self.dictTrainData[key][1]))

    def splitData(self, floatRatio):
        data = self.dictTrainData["Data"]
        X = data[0]
        Y = data[1]
        intDataSize = len(X)
        intValidationSize = int(intDataSize * floatRatio)
        return (X[intValidationSize:], Y[intValidationSize:]), (X[:intValidationSize], Y[:intValidationSize])

