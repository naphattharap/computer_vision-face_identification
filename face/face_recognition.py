#!/usr/bin/env python
__author__ = "Naphatthara P."
__version__ = "1.0.0"

"""
    Read model and predict ID of faces on camera.
"""
import cv2 as cv

# PATH_PHOTO_TRAIN = "dataset"

PATH_HAAR_CASCADE_FRONTAL_FACE = 'face/haarcascade/haarcascade_frontalface_default.xml'
PATH_PHOTO_STORAGE = "../dataset"
PATH_RECOGNIZER = "model_building/face_recognizer_model.yml"


class FaceIdentification:
    """
        Identify ID of face on camera.
    """

    def __init__(self, camera_window_name="camera"):
        self.recognizer = cv.face.LBPHFaceRecognizer_create()
        self.recognizer.read(PATH_RECOGNIZER)
        self.face_detector = cv.CascadeClassifier(PATH_HAAR_CASCADE_FRONTAL_FACE)
        self.camera = cv.VideoCapture(0)
        self.camera_window_name = camera_window_name
        cv.namedWindow(self.camera_window_name, cv.WINDOW_AUTOSIZE)

    def turnoff_camera(self):
        """
        Turn of the camera
        """
        if self.camera.isOpened():
            self.camera.release()
            cv.destroyAllWindows()

    def idenity_user(self):
        """
            Identify user
        """
        # Move camera to frontmost windows
        # os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')

        ret, img = self.camera.read()

        # Resize window
        img = cv.resize(img, (960, 540))

        # convert to grayscale image
        grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # bounding box of faces on camera
        faces = self.face_detector.detectMultiScale(grayscale, 1.3, 5)
        font = cv.FONT_HERSHEY_SIMPLEX
        for (x_pos, y_pos, width, height) in faces:
            cv.rectangle(img, (x_pos, y_pos), (x_pos + width, y_pos + height), (0, 255, 0), 2)
            # src    Sample image to get a prediction from.
            # label    The predicted label for the given image.
            # confidence    Associated confidence (e.g. distance) for the predicted label.
            label, confidence = self.recognizer.predict(grayscale[y_pos:y_pos + height, \
                                                                  x_pos:x_pos + width])
            # print("id/confidence: ", id, "/", confidence)

            # Add text for showing ID
            cv.putText(img, "ID:" + str(label), (x_pos + 5, y_pos - 5), \
                       font, 1, (252, 186, 3), 2)
            cv.putText(img, str(confidence), (x_pos + 5, y_pos + height - 5), \
                       font, 1, (255, 255, 0), 2)

        cv.imshow(self.camera_window_name, img)

