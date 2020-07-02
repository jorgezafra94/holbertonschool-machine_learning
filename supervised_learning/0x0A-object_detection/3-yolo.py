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
        multi = []
        for confidence, clase in zip(box_confidences, box_class_probs):
            multi.append(confidence * clase)

        # processing index_classes and turn them into a vector
        # awesome trick of reshape(-1)
        index_class = [np.argmax(elem, axis=-1) for elem in multi]
        index_class = [elem.reshape(-1) for elem in index_class]
        index_class = np.concatenate(index_class)
        # processing score_class and turn them into vector
        score_class = [np.max(elem, axis=-1) for elem in multi]
        score_class = [elem.reshape(-1) for elem in score_class]
        score_class = np.concatenate(score_class)

        # mask is going to return a list with the positions
        # that fulfill the condition
        mask = np.where(score_class >= self.class_t)

        box_class = index_class[mask]
        box_score = score_class[mask]

        # processing the boxes, turn them into matrix of (?, 4)
        filter_box = [elem.reshape(-1, 4) for elem in boxes]
        filter_box = np.concatenate(filter_box)
        filter_box = filter_box[mask]

        return (filter_box, box_class, box_score)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        * filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of
            the filtered bounding boxes:
        * box_classes: a numpy.ndarray of shape (?,) containing the class
            number for the class that filtered_boxes predicts, respectively
        * box_scores: a numpy.ndarray of shape (?) containing the box scores
            for each box in filtered_boxes, respectively
        Returns a tuple of
            (box_predictions, predicted_box_classes, predicted_box_scores):
        * box_predictions: a numpy.ndarray of shape (?, 4) containing all
            of the predicted bounding boxes ordered by class and box score
        * predicted_box_classes: a numpy.ndarray of shape (?,) containing the
            class number for box_predictions ordered by class and box score,
            respectively
        * predicted_box_scores: a numpy.ndarray of shape (?) containing the box
            scores for box_predictions ordered by class and box score,
            respectively
        """
        #  nms_t: the IOU threshold for non - max suppression

        box_predictions = []
        pred_box_classes = []
        pred_box_scores = []
        unicos = np.unique(box_classes)
        for i in unicos:
            # get ids per class
            idx = np.where(box_classes == i)

            # getting infor related to that indexes
            filters = filtered_boxes[idx]
            scores = box_scores[idx]
            classes = box_classes[idx]
            keep = self.IoU(filters, self.nms_t, scores)

            # keep are the indexes to keep alive
            # so we have to filter the new info
            final_filters = filters[keep]
            final_scores = scores[keep]
            final_classes = classes[keep]

            box_predictions.append(final_filters)
            pred_box_classes.append(final_classes)
            pred_box_scores.append(final_scores)

        filtered_boxes = np.concatenate(box_predictions, axis=0)
        box_scores = np.concatenate(pred_box_scores, axis=0)
        box_classes = np.concatenate(pred_box_classes, axis=0)

        return (filtered_boxes, box_classes, box_scores)

    def IoU(self, bc, thresh, scores):
        """
        Here we realize all the process of the IoU calculation
        and return the best boxes
        """
        x1 = bc[:, 0]
        y1 = bc[:, 1]
        x2 = bc[:, 2]
        y2 = bc[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep
