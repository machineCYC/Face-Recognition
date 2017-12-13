# Face-Recognition
This project focused on the methodology of PCA (Principal component analysis) and SVM (Support vector machine). We implemented the face detection and extraction using OpenCV, data processing through PCA and face recognition based on SVM algorithm, finally performance evaluation.

## Requirements

* Python 3.5
* PIL
* cv2 (opencv2)
* Numpy
* Scikit-learn
* Matplotlib
 
## Datasets

* [Yale Facedatabase A](http://vision.ucsd.edu/content/yale-face-database), also known as Yalefaces. The AT&T Facedatabase is good for initial tests, but it’s a fairly easy database. The Eigenfaces method already has a 97% recognition rate on it, so you won’t see any great improvements with other algorithms. 
* The Yale Facedatabase A (also known as Yalefaces) is a more appropriate dataset for initial experiments, because the recognition problem is harder. The database consists of 15 people (14 male, 1 female) each with 11 grayscale images sized 320 $\times$ 243 pixel. There are changes in the light conditions (center light, left light, right light), facial expressions (happy, normal, sad, sleepy, surprised, wink) and glasses (glasses, no-glasses).

## Methodology

The problem with the image representation we are given is its high dimensionality. Two-dimensional p $\times$ q grayscale images span a m = pq-dimensional vector space. Therefore we through PCA to reduce the dimension of the images. Finally, use transform data and labels to be the input of classification algorithm SVM. The predicting part, use the already training PCA to transform the test data, then put it into the SVM to predict.

* [Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)

The Principal Component Analysis (PCA), which is the core of the Eigenfaces method, finds the directions with the greatest variance in the data, called principal components.

<!-- Let $X$ is a data matrix, with column-wise zero empirical mean, where each of the N rows represents a image, and each of the M columns gives a particular kind of features, -->


* [Support vector machine](https://en.wikipedia.org/wiki/Support_vector_machine)

## Reference
http://www.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf

