import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform


strProjectPath = os.path.dirname(__file__)
strRAWDataPath = os.path.join(strProjectPath, "01-RAWData")

strFaceDataPath = os.path.join(strRAWDataPath, "Aberdeen")
strFaceData = [os.path.join(strFaceDataPath, img) for img in os.listdir(strFaceDataPath)]
strOutputPath = os.path.join(strProjectPath, "Output")

class prepareImage():
    def __init__(self, intSize):
        self.listImages = []
        self.listImagesVector = []
        self.intSize = intSize

    def doResizeImage(self):
        for image in strFaceData:
            arrayImage = io.imread(image, as_grey=False)/255.0
            self.listImages.append(transform.resize(arrayImage, (self.intSize, self.intSize, 3)))

    def doFlattenImage(self):
        for image in self.listImages:
            self.listImagesVector.append(image.flatten())

        
class PCA():
    def __init__(self, intImageSize, intTopEigenNum, intReConImgNum):
        self.intImageSize = intImageSize
        self.intTopEigenNum = intTopEigenNum
        self.intReConImgNum = intReConImgNum

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

        fig = plt.figure(figsize=(12, 2))
        for i in range(self.intTopEigenNum):
            EigenFaces = np.array(arrayEigenVectors[:, 0:self.intTopEigenNum][:, i])
            EigenFaces -= np.min(EigenFaces)
            EigenFaces /= np.max(EigenFaces)
            EigenFaces = (EigenFaces * 255).astype(np.uint8)

            ax = fig.add_subplot(1, self.intTopEigenNum, i+1)
            ax.imshow(EigenFaces.reshape((self.intImageSize, self.intImageSize, 3)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        plt.savefig(os.path.join(strOutputPath, "Top{}EigenFaces".format(self.intTopEigenNum)))
        return arrayEigenValues, arrayEigenVectors

    def calculateEigenValuePercentage(self, arrayEigenValues):
        arrayEigenValuesPercentage = arrayEigenValues / np.sum(arrayEigenValues)
        return np.round(arrayEigenValuesPercentage, 1)

    def reconstructImages(self, arrayImagesVector, arrayEigenVectors, arrayImageMean):
        arrayRandomIndex = np.random.randint(0, 415, size=self.intReConImgNum)
        arrayRandomImagesVector = arrayImagesVector[arrayRandomIndex]

        arrayTopNEigenVectors = arrayEigenVectors[:, 0:self.intTopEigenNum]
        arrayWeight = np.dot(arrayRandomImagesVector, arrayTopNEigenVectors)
        arrayReconImagesVectors = np.dot(arrayWeight, np.transpose(arrayTopNEigenVectors))

        fig = plt.figure(figsize=(12, 4))
        for i in range(self.intReConImgNum):
            ReconImage = arrayReconImagesVectors[i]
            ReconImage = ((ReconImage + arrayImageMean)*255).astype(np.uint8)

            ax = fig.add_subplot(2, 4, i+1)
            ax.imshow(ReconImage.reshape((self.intImageSize, self.intImageSize, 3)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
            plt.title("ReconImage{}".format(arrayRandomIndex[i]))

            arrayOriginImage = arrayImagesVector[arrayRandomIndex[i]]
            arrayOriginImage = ((arrayOriginImage + arrayImageMean)*255).astype(np.uint8)

            ax = fig.add_subplot(2, 4, i+5)
            ax.imshow(arrayOriginImage.reshape((self.intImageSize, self.intImageSize, 3)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
            plt.title("Origin{}".format(arrayRandomIndex[i]))
        plt.savefig(os.path.join(strOutputPath, "Recon{}RandomImg{}Eigen".format(self.intReConImgNum, self.intTopEigenNum)))
        return arrayReconImagesVectors


if __name__ == "__main__":
    processing = prepareImage(intSize=64)
    processing.doResizeImage()
    processing.doFlattenImage()

    pca = PCA(intImageSize=64, intTopEigenNum=5, intReConImgNum=4)
    arrayImagesVector = np.array(processing.listImagesVector)

    arrayImageMean, arrayImagesVector = pca.getZeroMeanImage(arrayImagesVector)
    arrayEigenValues, arrayEigenVectors = pca.getEigenValueVector(arrayImagesVector)
    arrayEigenValuesPercentage = pca.calculateEigenValuePercentage(arrayEigenValues)
    # print(arrayEigenValuesPercentage)

    pca.reconstructImages(arrayImagesVector, arrayEigenVectors, arrayImageMean)
