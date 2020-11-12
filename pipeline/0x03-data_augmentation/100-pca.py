#!/usr/bin/env python3
"""
pca from Alexnet
"""
import tensorflow as tf
import numpy as np


def pca_color(img, alpha):
    """
    * performs PCA color augmentation as described in the AlexNet paper
    * param img: 3D tf.Tensor containing the image to change
    * param alpha: tuple of length 3 containing the amount that each channel
    Return the augmented image
    """
    original = tf.keras.preprocessing.image.img_to_array(img)

    first = original.shape[0]
    sec = original.shape[1]
    # first you need to unroll the image into a nx3 where 3 is the
    # no. of colour channels
    renorm_image = np.reshape(original, (first * sec, 3))

    # Before applying PCA you must normalize the data in each column
    # separately as we will be applying PCA column-wise

    mean = np.mean(renorm_image, axis=0)
    std = np.std(renorm_image, axis=0)
    renorm_image = renorm_image.astype('float32')
    renorm_image -= np.mean(renorm_image, axis=0)
    renorm_image /= np.std(renorm_image, axis=0)

    # finding the co-variance matrix for computing the eigen values
    # and eigen vectors
    cov = np.cov(renorm_image, rowvar=False)

    # finding the eigen values lambdas and the vectors p
    lambdas, p = np.linalg.eig(cov)

    # delta here represents the value which will be added to the
    # re_norm image
    delta = np.dot(p, alpha * lambdas)

    pca_augmentation = renorm_image + delta
    pca_color_image = pca_augmentation * std + mean

    minimo = np.minimum(pca_color_image, 255)
    pca_color_image = np.maximum(minimo, 0).astype('uint8')

    pca_color_image = np.ravel(pca_color_image)
    pca_color_image = pca_color_image.reshape((first, sec, 3))

    result = tf.keras.preprocessing.image.array_to_img(pca_color_image)
    return result
