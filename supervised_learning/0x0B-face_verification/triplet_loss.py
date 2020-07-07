#!/usr/bin/env python3
"""
Triple Loss
"""

import numpy as np
import tensorflow.keras as K
from tensorflow.keras.layers import Layer


class TripletLoss(Layer):
    """
    class Tripletloss
    """

    def __init__(self, alpha, **kwargs):
        """
        * alpha is the alpha value used to calculate the triplet loss
        * sets the public instance attribute alpha
        """
        super().__init__(**kwargs)
        self.alpha = alpha