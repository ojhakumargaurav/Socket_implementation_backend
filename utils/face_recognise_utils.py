import numpy as np
import imutils
import pickle
import cv2
import os
from .video_utils import VideoUtils
'''
this file has a class which will load the image recognizer in memory using class method which will run only once.
then we can run the recognize function to return the image including the text around it.

'''


class RecognizeFace(VideoUtils):
    def __init__(self, *args, **kwargs):
        if(RecognizeFace.detector == None or RecognizeFace.embedder == None):
            RecognizeFace.load_models()
        VideoUtils.__init__(self, *args, **kwargs)

    detector = None
    embedder = None
    THRESHOLD = 0.3

    @classmethod
    def load_models(cls):
        # load the model and label encoder
        caffemodel_path = os.path.join(
            *["Pre_trained_models", "Face_recognition_models", "res10_300x300_ssd_iter_140000.caffemodel"])
        deploy_prototxt_path = os.path.sep.join(
            ["Pre_trained_models", "Face_recognition_models", "deploy.prototxt"])
        cls.detector = cv2.dnn.readNetFromCaffe(
            deploy_prototxt_path, caffemodel_path)
        # load our serialized face embedding model from disk
        embedder_path = os.path.join(
            *["Pre_trained_models", "Face_recognition_models", "openface.nn4.small2.v1.t7"])
        print("[INFO] loading face recognizer...")
        cls.embedder = cv2.dnn.readNetFromTorch(embedder_path)

    def process_image(self, image, is_image_path=False):
        # this function is used to recognize the face is wearing the mask correctly or not
        if(is_image_path):
            image = cv2.imread(image)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (100, 120)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        RecognizeFace.detector.setInput(imageBlob)
        detections = RecognizeFace.detector.forward()
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections
            if confidence > RecognizeFace.THRESHOLD:
                # compute the (x, y)-coordinates of the bounding box for the
                # face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # extract the face ROI
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
        return image
