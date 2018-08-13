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
    def __init__(self, int_Image_Size):
        self.list_Images_Vector = []
        self.int_Image_Size = int_Image_Size

    def resizeImage(self):
        """
        This function can resize the image. 
        In this process, it will do normalize for each image then resize the image.
        """
        for image in strFaceData:
            arrayImage = io.imread(image, as_grey=False)
            arrayResizeImage = transform.resize(arrayImage, (self.int_Image_Size, self.int_Image_Size, 3))
            self.list_Images_Vector.append(arrayResizeImage.flatten())

        
class PCA(object):
    def __init__(self, int_TopEigen_Num=4):
        self.int_TopEigen_Num = int_TopEigen_Num

        self.array_Mean_Image = None
        self.array_EigenValues = None
        self.array_EigenVectors = None

        self.array_EigenValues_Ratio = None

    def fit(self, X):
        array_ZeroMean_Images = self.get_ZeroMean_Image(X) # (N, D)

        # array_Cov = np.dot(array_X.T, array_X) # (D, D)
        # self.array_EigenValues, self.array_EigenVectors = np.linalg.eig(array_Cov) # (N,) (N, N)
        # print("fit finish")
        array_Cov = np.dot(array_ZeroMean_Images, array_ZeroMean_Images.T) # (N, N)
        self.array_EigenValues, array_EigenVectors = np.linalg.eig(array_Cov) # (N,) (N, N)
        self.array_EigenVectors = np.dot(array_ZeroMean_Images.T, array_EigenVectors) # (D, N)

    def transform(self, X, int_Num_Image=4):

        array_Random_Index = np.random.randint(0, 415, size=int_Num_Image)
        array_Random_Images_Vector = self.get_ZeroMean_Image(X)[array_Random_Index] # (n, D)

        array_TopEigen_Vectors = self.array_EigenVectors[:, 0:self.int_TopEigen_Num] # (D, K)

        array_projecttion_matrix = np.dot(array_TopEigen_Vectors, array_TopEigen_Vectors.T) # (D, D)

        # array_Weight = np.dot(array_Random_Images_Vector, array_TopEigen_Vectors) # (n, K)
        # array_Recon_Images_Vectors = np.dot(array_Weight, np.transpose(array_TopEigen_Vectors)) # (n, D)
        array_Recon_Images_Vectors = np.dot(array_Random_Images_Vector, array_projecttion_matrix) # (n, D))
        return array_Recon_Images_Vectors, array_Random_Index

    def cal_EigenValues_Ratio(self):
        array_EigenValues_Ratio = self.array_EigenValues / np.sum(self.array_EigenValues)
        return np.round(array_EigenValues_Ratio, 3)


    def get_ZeroMean_Image(self, X):
        """
        X: A list have N array, each array have D dim. ex:[array(d1, ...,dD), ..., array(d1, ...,dD)] 
        """
        self.array_Mean_Image = np.mean(X, axis=0) #(D,)
        array_ZeroMean_Images = np.array(X) - self.array_Mean_Image # (N, D)
        return array_ZeroMean_Images


def get_Img_Clip(img):
    img -= np.min(img)
    img /= np.max(img)
    img = (img * 255).astype(np.uint8)
    img = np.reshape(img, (128, 128, 3))
    return img


if __name__ == "__main__":
    processing = prepareImage(int_Image_Size=128)
    processing.resizeImage()
    list_Images_Vector = processing.list_Images_Vector # NxD

    pca = PCA(int_TopEigen_Num=4)
    pca.fit(X=list_Images_Vector)
    array_TopK_EigenVectors = pca.array_EigenVectors[:, 0:4] # (D, 4)

    # plot avg face
    plt.imshow(pca.array_Mean_Image.reshape(128, 128, 3))
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.savefig(os.path.join(strOutputPath, "AvgFace"))

    # plot top 4 eigen face
    fig = plt.figure(figsize=(12, 2))
    for i in range(4):
        ax = fig.add_subplot(1, 4, i+1)
        # let each value in 0 ~ 255
        array_Topi_EigenVectors = array_TopK_EigenVectors[:, i]

        # ax.imshow(array_Topi_EigenVectors.reshape((128, 128, 3)))
        ax.imshow(get_Img_Clip(array_Topi_EigenVectors))
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    plt.savefig(os.path.join(strOutputPath, "Top{}EigenFaces".format(4)))

    array_Recon_Images_Vectors, array_Random_Index = pca.transform(X=list_Images_Vector, int_Num_Image=4)

    # plot 4 random reconstruct images
    fig = plt.figure(figsize=(12, 4))
    for i in range(4):

        ReconImage = array_Recon_Images_Vectors[i]
        # ReconImage += pca.array_Mean_Image

        ReconImage -= np.min(ReconImage)
        ReconImage /= np.max(ReconImage)

        ReconImage += pca.array_Mean_Image

        ReconImage = (ReconImage * 255).astype(np.uint8)
        ReconImage = np.reshape(ReconImage, (128, 128, 3))

        ax = fig.add_subplot(2, 4, i+1)
        # ax.imshow(get_Img_Clip(ReconImage))
        ax.imshow(ReconImage)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
        plt.title("ReconImage{}".format(array_Random_Index[i]))

        arrayOriginImage = list_Images_Vector[array_Random_Index[i]]

        ax = fig.add_subplot(2, 4, i+5)
        ax.imshow(arrayOriginImage.reshape((128, 128, 3)))
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
        plt.title("Origin{}".format(array_Random_Index[i]))
    plt.savefig(os.path.join(strOutputPath, "Recon{}RandomImg{}Eigen".format(4, 4)))

