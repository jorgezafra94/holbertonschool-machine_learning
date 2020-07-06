#!/usr/bin/env python3
"""
utilities for face detection
"""

import tensorflow as tf
import numpy as np
import glob
import cv2
import csv

def load_images(images_path, as_array=True):
    """
    * images_path is the path to a directory from which to load images
    * as_array is a boolean indicating whether the images should be
        loaded as one numpy.ndarray
    * If True, the images should be loaded as a numpy.ndarray of
        shape (m, h, w, c) where:
        -  m is the number of images
        -  h, w, and c are the height, width, and number of channels
            of all images, respectively
    * If False, the images should be loaded as a list of individual
        numpy.ndarrays
    * All images should be loaded in RGB format
    * The images should be loaded in alphabetical order by filename
    Returns: images, filenames
    * images is either a list/numpy.ndarray of all images
    * filenames is a list of the filenames associated with each image in images
    """
    list_names = []
    list_img = []

    path_list = glob.glob(images_path + '/*', recursive=False)
    path_list.sort()

    for image_name in path_list:
        # ************** LINUX ****************
        name = image_name.split('/')[-1]
        # ************* WINDOWS **************
        name = name.split('\\')[-1]
        list_names.append(name)

    for image_name in path_list:
        # ******************* WINDOWS & LINUX ***************************
        image = cv2.imdecode(np.fromfile(image_name, np.uint8),
                             cv2.IMREAD_UNCHANGED)
        # ******************* LINUX ******************************
        # imagen = cv2.imread(image_name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        list_img.append(image_rgb)

    if as_array is True:
        list_img = np.array(list_img)

    return(list_img, list_names)

def load_csv(csv_path, params={}):
    """
    * csv_path is the path to the csv to load
    * params are the parameters to load the csv with
    Returns: a list of lists representing the contents
        found in csv_path
    """
    csv_content = []
    with open(csv_path, encoding='utf-8') as fd:
        obj = csv.reader(fd, params)
        for line in obj:
            csv_content.append(line)

    return csv_content