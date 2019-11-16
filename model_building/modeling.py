#!/usr/bin/env python

__author__ = "Naphatthara P."
__version__ = "1.0.0"

import os
import cv2 as cv
import numpy as np
from PIL import Image

PATH_PHOTO_TRAIN = "../dataset/"
PATH_SAVE_MODEL = "face_recognizer_model.yml"


class FaceRecognizerModeling:
    
    def get_dataset(self, root_folder_path):
        # r=root, d=directories, f = files
        img_files = []
        ids = []
        for r, d, f in os.walk(PATH_PHOTO_TRAIN):
            for file in f:
                if '.jpg' in file:
                    id = os.path.basename(r)
                    print(id, "-", file)
                    ids.append(id)
                    
                    # the image is grayscale, so no need to convert before training
#                     img = Image.open(file)
                    img_files.append(os.path.join(r, file))
                    
        faces = []
        idx = 0
        for file in img_files:
                img = Image.open(file)
                img_numpy = np.array(img, 'uint8')
                faces.append(img_numpy)
                print("file: ", file)
                
        return faces, ids
        
    def train(self):
        recognizer = cv.face.LBPHFaceRecognizer_create()
        faces, ids = self.get_dataset(PATH_PHOTO_TRAIN)
        print(faces, ids)
        ids_int = list(map(int, ids))
        print(ids_int)
        recognizer.train(faces, np.array(ids_int))
        recognizer.save(PATH_SAVE_MODEL)
        
    def load_model(self):
        recognizer = cv.face.LBPHFaceRecognizer_create()
        recognizer.read(PATH_SAVE_MODEL)
        return recognizer


rec = FaceRecognizerModeling()
rec.train()
print(rec.load_model())
print
