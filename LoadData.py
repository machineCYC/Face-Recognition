import os, cv2
from PIL import Image
import numpy as np

def detectFace(image):
    # load face Classifier
    faceCascade = cv2.CascadeClassifier("Face-Recognition\\classifier\\data\\haarcascades\\haarcascade_frontalface_default.xml")

    face_w = 140
    face_h = 140
    faces = faceCascade.detectMultiScale(image) 
    # a list contain Top-Left x pixel value, Top-Left y pixel value, Width of rectangle, Height of rectangle.
    haveFace = (len(faces) > 0)
    if(haveFace):
        for (x, y, w, h) in faces:
            face = image[y:y + h, x:x + w]
            resize_face = cv2.resize(face, (face_w, face_h)) # (face_w, face_h)

            # cv2.imshow("show face", resize_face)
            # cv2.waitKey(100)
        return resize_face     
    else:
        return "upload a image with face"     


def getImageMatrixandLabels(image_paths):
    images = []
    labels = []
    for image_path in image_paths:
        image = np.array(Image.open(image_path), "uint8") # (243, 320)
        # cv2.imshow("show image", image)
        # cv2.waitKey(100)
        
        resize_face = detectFace(image)
        images.append(resize_face.flatten())
        labels.append(int(image_path.split("subject")[1].split(".")[0]))

    return images, labels
