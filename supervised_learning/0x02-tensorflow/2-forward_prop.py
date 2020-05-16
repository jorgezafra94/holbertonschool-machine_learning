#!/usr/bin/env python3
"""
forward propagation tensorflow
"""

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    x is the placeholder for the input data
    layer_sizes is a list containing the number
          of nodes in each layer of the network
    activations is a list containing the activation
           functions for each layer of the network
    Returns: the prediction of the network in tensor form
    For this function, you should import your create_layer
    function with create_layer = __import__('1-create_layer').create_layer
    """
    A = create_layer(x, layer_sizes[0], activations[0])
    
    for i in range(1, len(layer_sizes)):    
        A = create_layer(A, layer_sizes[i], activations[i])

    return A
