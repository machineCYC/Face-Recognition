import os, cv2
from PIL import Image
import numpy as np

def getImageMatrixandLabels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith("normal") or f.endswith("rightlight") or f.endswith("leftlight")]

    face_w = 140
    face_h = 140
    # load face Classifier
    faceCascade = cv2.CascadeClassifier("Face-Recognition\\classifier\\data\\haarcascades\\haarcascade_frontalface_default.xml")

    images = []
    labels = []
    for image_path in image_paths:
        image = np.array(Image.open(image_path), "uint8") # (243, 320)
        # cv2.imshow("show image", image)
        # cv2.waitKey(100)

        faces = faceCascade.detectMultiScale(image)
        for (x, y, w, h) in faces:
            face = image[y:y + h, x:x + w]
            resize_face = cv2.resize(face, (face_w, face_h)) # (face_w, face_h)

            images.append(resize_face.flatten())
            labels.append(int(image_path.split("subject")[1].split(".")[0]))

            # cv2.imshow("show face", resize_face)
            # cv2.waitKey(100)
    return images, labels
