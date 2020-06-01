#!/usr/bin/env python3
"""
Save configuration
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    network is the model whose configuration should be saved
    filename is the path of the file that the configuration should be saved to
    Returns: None
    """
    with open(filename, "w") as fd:
        json_model = network.to_json()
        fd.write(json_model)
    return None


def load_config(filename):
    """
    filename is the path of the file containing the models configuration
             in JSON format
    Returns: the loaded model
    """
    with open(filename, "r") as fd:
        model_json = fd.read()
        model = K.models.model_from_json(model_json)
    return model
