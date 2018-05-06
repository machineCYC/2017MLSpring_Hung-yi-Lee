import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform

# https://blog.csdn.net/u010006643/article/details/46417127
strProjectPath = os.path.dirname(__file__)
strRAWDataPath = os.path.join(strProjectPath, "01-RAWData")

strFaceDataPath = os.path.join(strRAWDataPath, "Aberdeen")
strFaceData = [os.path.join(strFaceDataPath, img) for img in os.listdir(strFaceDataPath)]

class prepareImage():
    def __init__(self):
        self.listImages = []
        self.listImagesVector = []

    def doResizeImage(self, intSize):
        for image in strFaceData:
            arrayImage = io.imread(image, as_grey=False)/255.0
            self.listImages.append(transform.resize(arrayImage, (intSize, intSize, 3)))

    def doFlattenImage(self):
        for image in self.listImages:
            self.listImagesVector.append(image.flatten())

        
class PCA():
    def __init__(self):
        self.aaa = None

    def getZeroMeanImage(self, arrayImagesVector): 
        arrayImageMean = np.mean(arrayImagesVector, axis=0)
        arrayImagesVector = arrayImagesVector - arrayImageMean
        return arrayImageMean, arrayImagesVector

    def getEigenValueVector(self, arrayImagesVector):
        arrayCov = np.dot(arrayImagesVector.T, arrayImagesVector)
        arrayEigenValues, arrayEigenVectors = np.linalg.eig(arrayCov)

        arrayEigenValuesIndex = np.argsort(arrayEigenValues)[::-1] 
        arrayEigenValues = arrayEigenValues[arrayEigenValuesIndex]
        arrayEigenVectors = arrayEigenVectors[:, arrayEigenValuesIndex]  
        return arrayEigenValues, arrayEigenVectors

    def calculateEigenValuePercentage(self, arrayEigenValues):
        arrayEigenValuesPercentage = arrayEigenValues / np.sum(arrayEigenValues)
        return arrayEigenValuesPercentage

    def reconstructImage(self):
        pass

    def plotTopNEigenFaces(self, arrayEigenVectors, N, intSize):
        fig = plt.figure(figsize=(12, 2))
        for i in range(N):
            EigenFace = np.array(arrayEigenVectors[:, 0:N][:, i])
            EigenFace -= np.min(EigenFace)
            EigenFace /= np.max(EigenFace)
            EigenFace = (EigenFace * 255).astype(np.uint8)

            ax = fig.add_subplot(1, 4, i+1)
            ax.imshow(EigenFace.reshape((intSize, intSize, 3)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        plt.show()






a = prepareImage()
a.doResizeImage(intSize=64)
a.doFlattenImage()

b = PCA()
arrayImagesVector = np.array(a.listImagesVector)

arrayImageMean, arrayImagesVector = b.getZeroMeanImage(arrayImagesVector)
arrayEigenValues, arrayEigenVectors = b.getEigenValueVector(arrayImagesVector)
arrayEigenValuesPercentage = b.calculateEigenValuePercentage(arrayEigenValues)
b.plotTopNEigenFaces(arrayEigenVectors, N=4, intSize=64)





arrayLowDimImagesVector = arrayImagesVector * arrayNthEigenVectors  
arrayReconImagesVector = (arrayLowDimImagesVector * arrayNthEigenVectors.T) + arrayImageMean  