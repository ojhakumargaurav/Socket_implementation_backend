
import numpy as np
import imutils
import time
import cv2
import os
import io
import tempfile


class VideoUtils(object):
    def process_image(self, image_frame):
        pass

    def process_video(self, video_file_blob):

        with tempfile.NamedTemporaryFile() as tmp:
            with open(f"{tmp.name}.webm", "wb") as file_opened:
                file_opened.write(video_file_blob)
                vs = cv2.VideoCapture(f"{tmp.name}.webm")
            writer = None
            (W, H) = (None, None)

            # try to determine the total number of frames in the video file
            try:
                prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                    else cv2.CAP_PROP_FRAME_COUNT
                total = vs.get(prop)
                print("[INFO] {} total frames in video".format(total))

            # an error occurred while trying to determine the total
            # number of frames in the video file
            except:
                print("[INFO] could not determine # of frames in video")
                print("[INFO] no approx. completion time can be provided")
                total = -1

            # loop over frames from the video file stream
            while True:
                # read the next frame from the file
                (grabbed, frame) = vs.read()

                # if the frame was not grabbed, then we have reached the end
                # of the stream
                if not grabbed:
                    break

                # if the frame dimensions are empty, grab them
                if W is None or H is None:
                    (H, W) = frame.shape[:2]
                start = time.time()
                frame = self.process_image(frame)
                end = time.time()
                # construct a blob from the input frame and then perform a forward
                # pass of the YOLO object detector, giving us our bounding boxes
                # and associated probabilities

                # check if the video writer is None
                if writer is None:
                    # initialize our video writer
                    fourcc = cv2.VideoWriter_fourcc(*'VP90')

                    file_name = "output.webm"
                    writer = cv2.VideoWriter(file_name, fourcc, 30,
                                             (frame.shape[1], frame.shape[0]), True)

                    # some information on processing single frame
                    if total > 0:
                        elap = (end - start)
                        print(
                            "[INFO] single frame took {:.4f} seconds".format(elap))
                        print("[INFO] estimated total time to finish: {:.4f}".format(
                            elap * total))

                    # write the output frame to disk

                writer.write(frame)

                # release the file pointers
            print("[INFO] cleaning up...")
            writer.release()
            bytes_available = None
            vs.release()
            with open(file_name, "rb") as bytes_read:
                bytes_available = bytes_read.read()
        return bytes_available
