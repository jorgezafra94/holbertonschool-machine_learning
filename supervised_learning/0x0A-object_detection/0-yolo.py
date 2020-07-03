#!/usr/bin/env python3
"""
Yolo for object recognition
"""

import tensorflow.keras as K


class Yolo:
    """
    * model_path is the path to where a Darknet Keras model is stored
    * classes_path is the path to where the list of class names used for
       the Darknet model, listed in order of index, can be found
    * class_t is a float representing the box score threshold for the
        initial filtering step
    * nms_t is a float representing the IOU threshold for non-max suppression
    * anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2) containing
        all of the anchor boxes:
        * outputs is the number of outputs (predictions) made by the Darknet
            model
        * anchor_boxes is the number of anchor boxes used for each prediction
    * 2 => [anchor_box_width, anchor_box_height]
    Public instance attributes:
    * model: the Darknet Keras model
    * class_names: a list of the class names for the model
    * class_t: the box score threshold for the initial filtering step
    * nms_t: the IOU threshold for non-max suppression
    * anchors: the anchor boxes
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        constructor
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as fd:
            all_classes = fd.read()
            all_classes = all_classes.split('\n')[:-1]
            self.class_names = all_classes

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
