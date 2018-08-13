import os
import numpy as np


strProjectPath = os.path.dirname(__file__)
strRAWDataPath = os.path.join(strProjectPath, "01-RAWData")

strFaceDataPath = os.path.join(strRAWDataPath, "all_sents.txt")
strOutputPath = os.path.join(strProjectPath, "Output")