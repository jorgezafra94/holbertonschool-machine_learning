#!/usr/bin/env python3
"""
Face Align
"""

import dlib
import cv2


class FaceAlign:
    """
    our FaceAlign
    """
    def __init__(self, shape_predictor_path):
        """
        * shape_predictor_path is the path to the dlib shape
            predictor model
        Sets the public instance attributes:
        * detector - contains dlibs default face detector
        * shape_predictor - contains the dlib.shape_predictor
        """
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """
        * This method detects a face in an image
        * image is a numpy.ndarray of rank 3 containing an image from which
          to detect a face
        Returns: a dlib.rectangle containing the boundary box for the face
          in the image, or None on failure
        * If multiple faces are detected, return the dlib.rectangle with the
          largest area
        * If no faces are detected, return a dlib.rectangle that is the same
          as the image
        """
        # we have to change the image to gray scale
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # we get the faces in the gray picture
        faces = self.detector(img_gray)

        # we have to go through all the faces
        box = [0, 0, 0, 0]
        large = 0
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            print(x1, y1, x2, y2)
            large_aux = (x2 - x1) * (y2 - y1)
            if (large_aux >= large):
                large = large_aux
                box = list((x1, y1, x2, y2))

        box_coordinates = tuple(box)
        return(dlib.rectangle(*box_coordinates))
