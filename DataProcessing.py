from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import os
import numpy as np

def plot_digits(instances, images_per_row=5, size=140,  **options):
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images)
    plt.imshow(image, cmap="gray", **options)

def mainPCA(images):
    if("PCA_model.pkl" in os.listdir("Face-Recognition/output")):
        # load pca
        pca = joblib.load("Face-Recognition/output/PCA_model.pkl") 

    else:
        # train pca
        pca = PCA(n_components=0.95, whiten=True, random_state=42)
        pca.fit(images)

        # save pca 
        joblib.dump(pca,"Face-Recognition/output/PCA_model.pkl")

        # plot the average face and save the image
        eigenMean = pca.mean_
        avgFace = eigenMean.reshape((140, 140)) # (140, 140)
        plt.figure(figsize=(10, 10))
        plt.imshow(avgFace, cmap="gray")
        plt.savefig("Face-Recognition/output/Average Face")
        
        # plot the eigenface image and save
        plt.figure(figsize=(10, 10))
        plot_digits(pca.components_, images_per_row=5, size=140)
        plt.savefig("Face-Recognition/output/Eigenfaces")

    return pca


