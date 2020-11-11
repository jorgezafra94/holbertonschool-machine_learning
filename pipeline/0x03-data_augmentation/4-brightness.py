#!/usr/bin/env python3
"""
Change Brightness of a 3d tensorflow image
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    * image is a 3D tf.Tensor containing
      the image to change
    * max_delta is the maximum amount the image
      should be brightened (or darkened)
    Returns the altered image
    """
    bright = tf.image.adjust_brightness(image, max_delta)
    return bright
