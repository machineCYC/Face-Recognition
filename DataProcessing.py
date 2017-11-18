from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import os

def mainPCA(images):
    if("PCA_model.pkl" in os.listdir("Face-Recognition/output")):
        # load pca
        pca = joblib.load("PCA_model.pkl") 

    else:
        # train pca
        pca = PCA(n_components=0.95, random_state=42)
        pca.fit(images)

        # save pca 
        joblib.dump(pca, "PCA_model.pkl")

        # plot the average face and save the image
        eigenMean = pca.mean_
        avgFace = eigenMean.reshape((140, 140)) # (140, 140)
        plt.imshow(avgFace, cmap="gray")
        plt.savefig("Face-Recognition/output/Average Face")

        # turn eigenvector to eigenface
        eigenfaces = pca.components_.reshape((pca.n_components_, 140, 140))
        
        # save eigenface image
        for i in range(pca.n_components_):
            plt.imshow(eigenfaces[i], cmap="gray")
            plt.savefig("Face-Recognition/output/eigenface" + str(i) + ".png")

    return pca


