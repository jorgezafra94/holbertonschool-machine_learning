#!/usr/bin/env python3
"""
change Hue of 3d image tensorflow
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    * image is a 3D tf.Tensor containing the image to change
    * delta is the amount the hue should change
    Returns the altered image
    """
    hue = tf.image.adjust_hue(image, delta)
    return hue
