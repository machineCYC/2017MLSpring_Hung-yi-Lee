import os, json
from keras.preprocessing.text import Tokenizer
import _pickle as pk
from keras.preprocessing.sequence import pad_sequences


strProjectFolder = os.path.dirname(os.path.dirname(__file__))
strRAWDataFolder = os.path.join(strProjectFolder, "01-RAWData")
strAPDataFolder = os.path.join(strProjectFolder, "02-APData")

class executeETL():
    def __init__(self):
        self.dictTrainData = {}

    def cleanData(self, strDataFile, boolLabel):
        listLabel = []
        listText = []
        with open(os.path.join(strRAWDataFolder, strDataFile), "r", encoding="utf8") as data:
            for d in data:
                if boolLabel:
                    listRow = d.split(" +++$+++ ")
                    listLabel.append(int(listRow[0]))
                    listText.append(listRow[1])
                else:
                    listText.append(d)

            if boolLabel:
                self.dictTrainData["LabelData"] = [listText, listLabel]
            else:
                self.dictTrainData["LabelData"] = [listText]

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
            self.dictTrainData[key][0] = pad_sequences(listSequence, maxlen=intSequenceLength)







ETL = executeETL()
ETL.cleanData(strDataFile="training_label.txt", boolLabel=True)
ETL.doTokenizer(intVocabSize=20000)
ETL.dictTrainData["LabelData"][0][0]
# ETL.saveTokenizer(strTokenizerFile="TokenizerDictionary")
ETL.loadTokenizer(strTokenizerFile="TokenizerDictionary")
ETL.convertWords2Sequence(intSequenceLength=40)
dictTrainData = ETL.dictTrainData







with open(os.path.join(strAPDataFolder, "training_label.json")) as json_data:
    d = json.load(json_data)
    json_data.close()