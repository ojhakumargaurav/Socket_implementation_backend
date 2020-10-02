# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import imutils
import time
import cv2
import os
import io
import tempfile
from utils.video_utils import VideoUtils


class YoloNetwork(VideoUtils):
    net = None
    ln = None
    LABELS = None
    COLORS = None

    @classmethod
    def load_yolo_components(cls):
        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join(
            ["Pre_trained_models", "yolo-coco", "coco.names"])
        cls.LABELS = open(labelsPath).read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        cls.COLORS = np.random.randint(0, 255, size=(len(YoloNetwork.LABELS), 3),
                                       dtype="uint8")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join(
            ["Pre_trained_models", "yolo-coco", "yolov4.weights"])
        configPath = os.path.sep.join(
            ["Pre_trained_models", "yolo-coco", "yolov4.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        # and determine only the *output* layer names that we need from YOLO
        print("[INFO] loading YOLO from disk...")
        cls.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        cls.ln = cls.net.getLayerNames()
        cls.ln = [cls.ln[i[0] - 1] for i in cls.net.getUnconnectedOutLayers()]

    def process_image(self, frame):
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        YoloNetwork.net.setInput(blob)
        layerOutputs = YoloNetwork.net.forward(YoloNetwork.ln)

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.5:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
                                0.3)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c)
                         for c in YoloNetwork.COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(YoloNetwork.LABELS[classIDs[i]],
                                           confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
