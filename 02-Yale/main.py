import os
from LoadData import getImageMatrixandLabels, detectFace
from DataProcessing import mainPCA, plot_digits
from Model import model
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(__file__), "yalefaces")
image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith("normal") or f.endswith("rightlight") or f.endswith("leftlight")]

# get the training images and label
images_train, labels_train = getImageMatrixandLabels(image_paths) # [45, 19600], [45, ]

# plot all face images of training set
plt.figure(figsize=(15, 15))
plot_digits(images_train, images_per_row=9, size=140)
plt.savefig("Face-Recognition/output/Train images")

# reduce the image dimension
pca = mainPCA(images_train)

# prepare the training data and label
X_train = pca.transform(images_train)
y_train = labels_train

# fit classification model
svc = model(X_train, y_train)

# testing the result
image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith("sad")]

# get the testing images and label
images_test, labels_test = getImageMatrixandLabels(image_paths)

# plot all face images of Testing set
plt.figure(figsize=(15, 15))
plot_digits(images_test, images_per_row=3, size=140)
plt.savefig("Face-Recognition/output/Test images")

# prepare the testing data and label
X_test = pca.transform(images_test)
y_test = labels_test

print(svc.predict(X_test))



