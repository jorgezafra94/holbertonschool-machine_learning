#!/usr/bin/env python3
"""
Yolo for object recognition
"""

import tensorflow.keras as K
import numpy as np


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
        load model, classes_names, class_t, nms_t, anchors
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as fd:
            all_classes = fd.read()
            all_classes = all_classes.split('\n')
            if len(all_classes[-1]) == 0:
                all_classes = all_classes[:-1]
            self.class_names = all_classes

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        * outputs is a list of numpy.ndarrays containing the predictions
          from the Darknet model for a single image: Each output will have
          the shape (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
           - grid_height & grid_width => the height and width of the grid used
             for the output
           - anchor_boxes => the number of anchor boxes used
           - 4 => (t_x, t_y, t_w, t_h)
           - 1 => box_confidence
           - classes => class probabilities for all classes
        * image_size is a numpy.ndarray containing the image’s original size
            [image_height, image_width]
        Returns a tuple of (boxes, box_confidences, box_class_probs):
        * boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
          anchor_boxes, 4)
          containing the processed boundary boxes for each output,
          respectively:
          - 4 => (x1, y1, x2, y2)
          - (x1, y1, x2, y2) should represent the boundary box relative to
            original image
        * box_confidences: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 1) containing the box
            confidences
            for each output, respectively
        * box_class_probs: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, classes) containing
            the box’s class
            probabilities for each output, respectively
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height = image_size[0]
        image_width = image_size[1]

        for i, output in enumerate(outputs):
            gh = output.shape[0]
            gw = output.shape[1]

            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]

            anchor = self.anchors[i]

            pw = anchor[:, 0]
            ph = anchor[:, 1]
            pw = pw.reshape(1, 1, len(pw))
            ph = ph.reshape(1, 1, len(ph))

            cx = np.tile(np.arange(gw), gh).reshape(gw, gw, 1)
            cy = np.tile(np.arange(gw),
                         gh).reshape(gh, gh, 1).T.reshape(gh, gh, 1)

            bx = self.sigmoid(t_x) + cx
            by = self.sigmoid(t_y) + cy

            bw = pw * np.exp(t_w)
            bh = ph * np.exp(t_h)

            # normalize
            bx = bx / gw
            by = by / gh
            bw = bw / self.model.input.shape[1].value
            bh = bh / self.model.input.shape[2].value

            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.zeros(output[:, :, :, :4].shape)
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2

            boxes.append(box)

            box_confidence = self.sigmoid(output[:, :, :, 4, np.newaxis])
            box_confidences.append(box_confidence)

            box_class = self.sigmoid(output[:, :, :, 5:])
            box_class_probs.append(box_class)

        return boxes, box_confidences, box_class_probs

    def sigmoid(self, x):
        """
        sigmoid function
        """
        return 1 / (1 + np.exp(-x))