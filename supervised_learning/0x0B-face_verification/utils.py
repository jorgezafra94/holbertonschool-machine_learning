#!/usr/bin/env python3
"""
utilities for face detection
"""

import tensorflow as tf
import numpy as np
import glob
import cv2
import csv
import os


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


def save_images(path, images, filenames):
    """
    * path is the path to the directory in which the
        images should be saved
    * images is a list/numpy.ndarray of images to save
    * filenames is a list of filenames of the images to save
    Returns: True on success and False on failure
    """
    if not os.path.exists(path):
        return False
    for i in range(len(images)):
        img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(path, filenames[i]), img_rgb)
    return True


def generate_triplets(images, filenames, triplet_names):
    """
    * images is a numpy.ndarray of shape (n, h, w, 3) containing
        the various images in the dataset
    * filenames is a list of length n containing the corresponding
        filenames for images
    * triplet_names is a list of lists where each sublist contains
        the filenames of an anchor, positive, and negative
        image, respectively
    Returns: a list [A, P, N]
    * A is a numpy.ndarray of shape (m, h, w, 3) containing
        the anchor images for all m triplets
    * P is a numpy.ndarray of shape (m, h, w, 3) containing
        the positive images for all m triplets
    * N is a numpy.ndarray of shape (m, h, w, 3) containing
        the negative images for all m triplets
    """
    _, h, w, c = images.shape
    list_A = []
    list_P = []
    list_N = []

    triple = [[i[0]+'.jpg', i[1]+'.jpg',
               i[2]+'.jpg'] for i in triplet_names]

    for elem in triple:
        flagA, flagP, flagN = (0, 0, 0)

        A_name, P_name, N_name = elem

        if A_name in filenames:
            flagA = 1
        if P_name in filenames:
            flagP = 1
        if N_name in filenames:
            flagN = 1

        if flagA and flagP and flagN:
            index_A = filenames.index(A_name)
            index_P = filenames.index(P_name)
            index_N = filenames.index(N_name)

            A = images[index_A]
            P = images[index_P]
            N = images[index_N]

            list_A.append(A)
            list_P.append(P)
            list_N.append(N)

    list_A = [elem.reshape(1, h, w, c) for elem in list_A]
    list_A = np.concatenate(list_A)

    list_P = [elem.reshape(1, h, w, c) for elem in list_P]
    list_P = np.concatenate(list_P)

    list_N = [elem.reshape(1, h, w, c) for elem in list_N]
    list_N = np.concatenate(list_N)

    print(list_A.shape, list_P.shape, list_N.shape)
    return (list_A, list_P, list_N)
