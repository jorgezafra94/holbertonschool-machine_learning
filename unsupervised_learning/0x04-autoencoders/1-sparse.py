#!/usr/bin/env python3
"""
Sparse Autoencoder
"""

import tensorflow.keras as keras


def sparse(input_dims, hidden_layers, latent_dims, lambtha):
    """
    * input_dims is an integer containing the dimensions of the model input
    * hidden_layers is a list containing the number of nodes for each hidden
      layer in the encoder, respectively
    * the hidden layers should be reversed for the decoder
    * lambtha is the regularization parameter used for L1 regularization on
      the encoded output
    * latent_dims is an integer containing the dimensions of the latent space
      representation

    Returns: encoder, decoder, auto
    * encoder is the encoder model
    * decoder is the decoder model
    * auto is the full autoencoder model
    * The autoencoder model should be compiled using adam optimization and
      binary cross-entropy loss
    * All layers should use a relu activation except for the last layer in the
      decoder, which should use sigmoid
    """
    reg = keras.regularizers.l1(lambtha)
    #  ****************************  ENCODER ****************************
    X_encode = keras.Input(shape=(input_dims,))
    hidden_e = keras.layers.Dense(units=hidden_layers[0], activation='relu',
                                  activity_regularizer=reg)
    Y_prev = hidden_e(X_encode)
    for i in range(1, len(hidden_layers)):
        hidden_e = keras.layers.Dense(units=hidden_layers[i],
                                      activation='relu',
                                      activity_regularizer=reg)
        Y_prev = hidden_e(Y_prev)

    latent_lay = keras.layers.Dense(units=latent_dims, activation='relu',
                                    activity_regularizer=reg)
    Y_encoded = latent_lay(Y_prev)
    encoder = keras.Model(inputs=X_encode, outputs=Y_encoded)
    # encoder.summary()

    # ****************************** DECODER *********************************
    X_decode = keras.Input(shape=(latent_dims,))
    hidden_d = keras.layers.Dense(units=hidden_layers[-1], activation='relu')
    Y_prev = hidden_d(X_decode)
    for j in range(len(hidden_layers) - 2, -1, -1):
        hidden_d = keras.layers.Dense(units=hidden_layers[j],
                                      activation='relu')
        Y_prev = hidden_d(Y_prev)
    last_layer = keras.layers.Dense(units=input_dims, activation='sigmoid')
    output = last_layer(Y_prev)
    decoder = keras.Model(inputs=X_decode, outputs=output)
    # decoder.summary()

    # ******************************* MODEL **********************************
    X_input = keras.Input(shape=(input_dims,))
    encode_output = encoder(X_input)
    decoder_output = decoder(encode_output)
    auto = keras.Model(inputs=X_input, outputs=decoder_output)
    # auto.summary()
    auto.compile(loss='binary_crossentropy', optimizer='adam')
    return (encoder, decoder, auto)
