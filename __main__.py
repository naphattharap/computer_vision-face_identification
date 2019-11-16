#!/usr/bin/env python

__author__ = "Naphatthara P."
__version__ = "1.0.0"

if __name__ == "__main__":
    from face.face_detection import FaceDetector
    from face.face_recognition import FaceIdentification
    import cv2 as cv
    import os

    # Create photo storage folder if it does not exist.
    PATH_PHOTO_STORAGE = "dataset"

    if not os.path.exists(PATH_PHOTO_STORAGE):
        os.mkdir(PATH_PHOTO_STORAGE)

    choice = input("Press [1] Face Detection, [2] Face Identification: ")

    if choice == '1':
        # Enter user id 
        print("Press 'ESC' for exist, 's' for saving image.")
        uid = input("Enter User ID:")

        # Init detector
        detector = FaceDetector()

        # Detect face and show in bounding box
        # Press 'ESC' for exiting video
        # Press 's' (115) for saving detected face

        while True:
            img, grayscale, faces_bbx = detector.detect_face()
            k = cv.waitKey(100) & 0xff
            # print(k)
            if k == 27:
                detector.turnoff_camera()
                break
            if k == 115:
                # Set image to dataset folder
                detector.save_image(uid, grayscale, faces_bbx, PATH_PHOTO_STORAGE)

    elif choice == '2':

        idenfier = FaceIdentification()

        print("Press 'ESC' to exit")
        while True:
            idenfier.idenity_user()
            k = cv.waitKey(100) & 0xff
            if k == 27:
                idenfier.turnoff_camera()
                break
