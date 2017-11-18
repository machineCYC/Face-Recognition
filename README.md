# Face-Recognition
The basic face recognition project

## Requirements

* Python 3.5
* PIL
* cv2 (opencv2)
* Numpy
* Scikit-learn
* Matplotlib

## Abstract

* This project focused on the methodology of PCA (Principal component analysis) and SVM (Support vector machine). We implemented the face detection and extraction using OpenCV, face recognition based on SVM algorithm, finally performance evaluation.
 
## Datasets

* [Yale Facedatabase A](http://vision.ucsd.edu/content/yale-face-database), also known as Yalefaces. The AT&T Facedatabase is good for initial tests, but it’s a fairly easy database. The Eigenfaces method already has a 97% recognition rate on it, so you won’t see any great improvements with other algorithms. 
* The Yale Facedatabase A (also known as Yalefaces) is a more appropriate dataset for initial experiments, because the recognition problem is harder. The database consists of 15 people (14 male, 1 female) each with 11 grayscale images sized 320 $\times$ 243 pixel. There are changes in the light conditions (center light, left light, right light), facial expressions (happy, normal, sad, sleepy, surprised, wink) and glasses (glasses, no-glasses).

## Methodology

The problem with the image representation we are given is its high dimensionality. Two-dimensional p $\times$ q grayscale images span a m = pq-dimensional vector space. We through PCA to reduce the dimension of the images

* [Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)

The Principal Component Analysis (PCA), which is the core of the Eigenfaces method, finds the directions with the greatest variance in the data, called principal components.

```flow
st=>start: Start
i=>inputoutput: 输入年份n
cond1=>condition: n能否被4整除？
cond2=>condition: n能否被100整除？
cond3=>condition: n能否被400整除？
o1=>inputoutput: 输出非闰年
o2=>inputoutput: 输出非闰年
o3=>inputoutput: 输出闰年
o4=>inputoutput: 输出闰年
e=>end
st->i->cond1
cond1(no)->o1->e
cond1(yes)->cond2
cond2(no)->o3->e
cond2(yes)->cond3
cond3(yes)->o2->e
cond3(no)->o4->e
```



* [Support vector machine](https://en.wikipedia.org/wiki/Support_vector_machine)



## 

