import os
from PIL import Image
import numpy as np
from LoadData import getImageMatrixandLabels, detectFace
from DataProcessing import mainPCA
from Model import model

path = "Face-Recognition/yalefaces/"

# get the training images and label
images, labels = getImageMatrixandLabels(path) # [45, 19600], [45, ]

# reduce the image dimension
pca = mainPCA(images)

# prepare the training data and label
X_train = pca.transform(images)
y_train = labels

# fit classification model
svc = model(X_train, y_train)

# testing the result
image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith("sad")]
for image_path in image_paths:
    image = np.array(Image.open(image_path), "uint8")
    resize_face = detectFace(image)
    X_test = pca.transform(resize_face.reshape(1, -1))
    print(svc.predict(X_test))


