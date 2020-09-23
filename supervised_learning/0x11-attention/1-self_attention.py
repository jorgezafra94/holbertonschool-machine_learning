#!/usr/bin/env python3
"""
I got it from:
https://towardsdatascience.com/implementing-neural-machine-
translation-with-attention-using-tensorflow-fc9c6f26155f

Attention layer
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    class SelfAttention
    """

    def __init__(self, units):
        """
        * units is an integer representing the number of hidden
          units in the alignment model
        * Sets the following public instance attributes:
        * W - a Dense layer with units units, to be applied to
          the previous decoder hidden state
        * U - a Dense layer with units units, to be applied to
          the encoder hidden states
        * V - a Dense layer with 1 units, to be applied to the
          tanh of the sum of the outputs of W and U
        """
        super().__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """
        * s_prev is a tensor of shape (batch, units) containing
          the previous decoder hidden state
        * hidden_states is a tensor of shape (batch, input_seq_len, units)
          containing the outputs of the encoder

        Returns: context, weights
        * context is a tensor of shape (batch, units) that contains the
          context vector for the decoder
        * weights is a tensor of shape (batch, input_seq_len, 1) that
          contains the attention weights
        """
        s_expanded = tf.expand_dims(input=s_prev, axis=1)
        first = self.U(s_expanded)
        second = self.W(hidden_states)
        score = self.V(tf.nn.tanh(first + second))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * s_expanded
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
