import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import time


strProjectPath = os.path.dirname(__file__)
strRAWDataPath = os.path.join(strProjectPath, "01-RAWData")

strFaceDataPath = os.path.join(strRAWDataPath, "Aberdeen")
strFaceData = [os.path.join(strFaceDataPath, img) for img in os.listdir(strFaceDataPath)]
strOutputPath = os.path.join(strProjectPath, "Output/pca")

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
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit(self, X):
        X = self._check_array(X)

        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        svd_star = time.time()
        U, S, V = np.linalg.svd(X.T, full_matrices=False)
        svd_end =  time.time()
        print("use {} s".format(svd_end - svd_star))

        self.S = S[0:self.n_components]
        self.U = U[:, 0:self.n_components]
        self.explained_variance_ratio_ = np.round(self.S / np.sum(self.S), 3)
        return
    
    def transform(self, X):
        X = self._check_array(X)
        X -= self.mean_

        Z = np.dot(X, self.U)
        return Z
    
    def inverse_transform(self, X):
        Z = self.transform(X)

        X_hat = np.dot(Z, self.U.T)
        return X_hat

    def _check_array(self, ndarray):
        ndarray = np.array(ndarray)
        if len(ndarray.shape) == 1: 
            ndarray = np.reshape(ndarray, (1, ndarray.shape[0]))
        return ndarray


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

    pca = PCA(n_components=4)
    pca.fit(X=list_Images_Vector)
    array_TopK_EigenVectors = pca.U # (D, 4)

    # plot avg face
    io.imsave(os.path.join(strOutputPath, "AvgFace.png"), pca.mean_.reshape(128, 128,3))
    # plt.imshow(pca.mean_.reshape(128, 128, 3))
    # plt.xticks(np.array([]))
    # plt.yticks(np.array([]))
    # plt.savefig(os.path.join(strOutputPath, "AvgFace"))

    # plot top 4 eigen face
    fig = plt.figure(figsize=(12, 2))
    for i in range(4):
        ax = fig.add_subplot(1, 4, i+1)
        # let each value in 0 ~ 255
        array_Topi_EigenVectors = array_TopK_EigenVectors[:, i]

        io.imsave(os.path.join(strOutputPath, "Top{}EigenFaces.png".format(i)), get_Img_Clip(array_Topi_EigenVectors))

    #     ax.imshow(get_Img_Clip(array_Topi_EigenVectors))
    #     plt.xticks(np.array([]))
    #     plt.yticks(np.array([]))
    #     plt.tight_layout()
    # plt.savefig(os.path.join(strOutputPath, "Top{}EigenFaces".format(4)))

    array_Random_Index = np.random.randint(0, 415, size=4)
    list_Random_Images_Vectors = [list_Images_Vector[i] for i in array_Random_Index]
    array_Recon_Images_Vectors = pca.inverse_transform(X=list_Random_Images_Vectors)

    # plot 4 random reconstruct images
    fig = plt.figure(figsize=(12, 4))
    for i in range(4):

        ReconImage = array_Recon_Images_Vectors[i]

        ReconImage += pca.mean_

        # ReconImage -= np.min(ReconImage)
        # ReconImage /= np.max(ReconImage)

        # ReconImage = (ReconImage * 255).astype(np.uint8)
        # ReconImage = np.reshape(ReconImage, (128, 128, 3))

        io.imsave(os.path.join(strOutputPath, "ReconImage{}.png".format(array_Random_Index[i])), get_Img_Clip(ReconImage))
        io.imsave(os.path.join(strOutputPath, "Origin{}.png".format(array_Random_Index[i])), list_Images_Vector[array_Random_Index[i]].reshape(128, 128,3))

    #     ax = fig.add_subplot(2, 4, i+1)
    #     ax.imshow(get_Img_Clip(ReconImage))
    #     # ax.imshow(ReconImage)
    #     plt.xticks(np.array([]))
    #     plt.yticks(np.array([]))
    #     plt.tight_layout()
    #     plt.title("ReconImage{}".format(array_Random_Index[i]))

    #     arrayOriginImage = list_Images_Vector[array_Random_Index[i]]

    #     ax = fig.add_subplot(2, 4, i+5)
    #     ax.imshow(arrayOriginImage.reshape((128, 128, 3)))
    #     plt.xticks(np.array([]))
    #     plt.yticks(np.array([]))
    #     plt.tight_layout()
    #     plt.title("Origin{}".format(array_Random_Index[i]))
    # plt.savefig(os.path.join(strOutputPath, "Recon{}RandomImg{}Eigen".format(4, 4)))

