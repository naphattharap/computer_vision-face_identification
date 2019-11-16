#!/usr/bin/env python
__author__ = "Naphatthara P."
__version__ = "1.0.0"

"""
In case of error: error: (-215:Assertion failed) !empty() in function 'detectMultiScale'

"""
import os
from datetime import datetime
import cv2 as cv

PATH_HAAR_CASCADE_FRONTAL_FACE = 'face/haarcascade/haarcascade_frontalface_default.xml'


class FaceDetector:
    """
    Detect face through input built-in camera and show the bounding box.
    """

    def __init__(self, camera_window_name="camera"):

        # Load haar cascade frontal face for face detector
        self.face_detector = cv.CascadeClassifier(PATH_HAAR_CASCADE_FRONTAL_FACE)
        """
            Open built-in camera
        """
        self.camera = cv.VideoCapture(0)
        self.camera_window_name = camera_window_name
        # CV_WINDOW_NORMAL enables you to resize the window,
        # whereas CV_WINDOW_AUTOSIZE adjusts automatically the window size
        # to fit the displayed image (see imshow() ),
        # and you cannot change the window size manually.
        cv.namedWindow(self.camera_window_name, cv.WINDOW_AUTOSIZE)

    def turnoff_camera(self):
        """
        Turn of the camera
        """
        if self.camera.isOpened():
            self.camera.release()
            cv.destroyAllWindows()

    def detect_face(self):
        """
        Detect face
        return
            image, grayscale images and face bounding boxes
        """

        # Move camera to frontmost windows
        # os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')

        ret, img = self.camera.read()

        # Resize window
        img = cv.resize(img, (960, 540))

        # convert to grayscale image
        grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # get bouding boxes of faces on camera
        faces_bbx = self.face_detector.detectMultiScale(grayscale, 1.3, 5)

        # Show bounding box
        for (x_pos, y_pos, width, height) in faces_bbx:
            cv.rectangle(img, (x_pos, y_pos), (x_pos + width, y_pos + height), (252, 186, 3), 2)
            cv.imshow(self.camera_window_name, img)

        return img, grayscale, faces_bbx

    @staticmethod
    def save_image(uid, img, bounding_boxes, storage_path):
        # Save the captured image into the storage folder
        id_photo_folder = storage_path + "/" + uid
        if not os.path.exists(id_photo_folder):
            os.mkdir(id_photo_folder)

        for (x, y, w, h) in bounding_boxes:
            # Generate text for current date time to be used as photo's ID.
            now = datetime.now()
            date_time = now.strftime("%Y%m%d%H%M%S")
            unique_path = id_photo_folder + "/" + date_time + ".jpg"
            print("Save detected faces to ", unique_path)
            cv.imwrite(unique_path, img[y:y + h, x:x + w])
