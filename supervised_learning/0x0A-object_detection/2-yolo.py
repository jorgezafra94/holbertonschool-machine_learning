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
            all_classes = all_classes.split()
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
        final_boxes = []
        confidence_boxes = []
        prop_boxes = []
        im_h, im_w = image_size
        for index, output in enumerate(outputs):
            grid_h, grid_w, anchor, total = output.shape
            # *************** Boxes *********************
            t_prediction = output[:, :, :, :4]
            boxes = np.zeros(t_prediction.shape)

            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]

            pw = self.anchors[:, :, 0]
            ph = self.anchors[:, :, 1]

            # creating matrices with respective dimensions, thay have to match
            # with the dims of t_x, t_y, t_w, t_h
            pw_actual = pw[index].reshape(1, 1, len(pw[index]))
            ph_actual = ph[index].reshape(1, 1, len(ph[index]))

            cx = np.tile(np.arange(0, grid_w), grid_h)
            cx = cx.reshape(grid_w, grid_w, 1)
            cy = np.tile(np.arange(0, grid_w), grid_h)
            cy = (cy.reshape(grid_h, grid_h).T).reshape(grid_h, grid_h, 1)

            # predictions
            bx = (1 / (1 + np.exp(-t_x))) + cx
            by = (1 / (1 + np.exp(-t_y))) + cy

            bw = np.exp(t_w) * pw_actual
            bh = np.exp(t_h) * ph_actual

            # Normalization
            bx = bx / grid_w
            by = by / grid_h

            bw = bw / self.model.input.shape[1].value
            bh = bh / self.model.input.shape[2].value

            # getting coordinates
            # (x1, y1) top left corner
            # (x2, y2) bottom right corner
            x1 = (bx - (bw / 2)) * im_w
            y1 = (by - (bh / 2)) * im_h
            x2 = (bx + (bw / 2)) * im_w
            y2 = (by + (bh / 2)) * im_h

            # fulling the boxes
            boxes[:, :, :, 0] = x1
            boxes[:, :, :, 1] = y1
            boxes[:, :, :, 2] = x2
            boxes[:, :, :, 3] = y2
            final_boxes.append(boxes)

            # *************** confidence boxes ****************
            t_c = output[:, :, :, 4]
            confidence = (1 / (1 + np.exp(-t_c)))
            confidence = confidence.reshape(grid_h, grid_w, anchor, 1)
            confidence_boxes.append(confidence)

            # ****** box class probabilities******************
            t_cprops = output[:, :, :, 5:]
            class_props = (1 / (1 + np.exp(-t_cprops)))
            prop_boxes.append(class_props)

        return (final_boxes, confidence_boxes, prop_boxes)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        * boxes: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 4) containing
            the processed boundary boxes for each output, respectively
        * box_confidences: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 1) containing the
            processed box confidences for each output, respectively
        * box_class_probs: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, classes) containing the
            processed box class probabilities for each output, respectively
        Returns a tuple of (filtered_boxes, box_classes, box_scores):
        * filtered_boxes: a numpy.ndarray of shape (?, 4) containing all
            of the filtered bounding boxes:
        * box_classes: a numpy.ndarray of shape (?,) containing the class
            number that each box in filtered_boxes predicts, respectively
        * box_scores: a numpy.ndarray of shape (?) containing the box scores
            for each box in filtered_boxes, respectively
        """
        box_scores_fin = []
        box_class_fin = []
        box_fil_fin = []
        for i in range(len(boxes)):
            conf = box_confidences[i]
            props = box_class_probs[i]
            filter = boxes[i]

            grid_h, grid_w, anchor, _ = filter.shape

            multi = conf[:, :, :] * props[:, :, :]
            class_position = np.argmax(multi, -1)
            class_scores = np.max(multi, -1)

            mask = np.where(class_scores[:, :, :] > self.class_t, 1, 0)

            box_scores = mask * class_scores
            box_classes = mask * class_position

            new_mask = np.repeat(mask, 4, axis=-1)
            new_mask = new_mask.reshape(grid_h, grid_w, anchor, 4)
            filtered_boxes = new_mask * filter

            for h in range(boxes[i].shape[0]):
                for w in range(boxes[i].shape[1]):
                    for a in range(boxes[i].shape[2]):
                        if(box_scores[h, w, a] != 0):
                            box_scores_fin.append(box_scores[h, w, a])
                        if (box_classes[h, w, a] != 0):
                            box_class_fin.append(box_classes[h, w, a])
                        if (filtered_boxes[h, w, a].sum() != 0):
                            box_fil_fin.append(filtered_boxes[h, w, a])

        score_fin = np.array(box_scores_fin)
        class_fin = np.array(box_class_fin)
        filter_fin = np.array(box_fil_fin)
        return (filter_fin, class_fin, score_fin)
