from LoadData import getImageMatrixandLabels
from DataProcessing import mainPCA

path = "Face-Recognition/yalefaces/"

images, labels = getImageMatrixandLabels(path) # [45, 19600], [45, ]

pca = mainPCA(images)

