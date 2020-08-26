#!/usr/bin/env python3
"""
Vanilla Autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    * input_dims is a tuple of integers containing the dimensions of
      the model input
    * filters is a list containing the number of filters for each
      convolutional layer in the encoder, respectively
    * the filters should be reversed for the decoder
    * latent_dims is a tuple of integers containing the dimensions of
      the latent space representation
    * Each convolution in the encoder should use a kernel size of (3, 3)
      with same padding and relu activation, followed by max pooling of
      size (2, 2)
    * Each convolution in the decoder, except for the last two, should
      use a filter size of (3, 3) with same padding and relu activation,
      followed by upsampling of size (2, 2)
    * The second to last convolution should instead use valid padding
    * The last convolution should have the same number of filters as the
      number of channels in input_dims with sigmoid activation and no
      upsampling

    Returns: encoder, decoder, auto
    * encoder is the encoder model
    * decoder is the decoder model
    * auto is the full autoencoder model
    * The autoencoder model should be compiled using adam optimization and
      binary cross-entropy loss
    """
    #  ****************************  ENCODER ****************************
    X_encode = keras.Input(shape=input_dims)

    conv_e = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                                 padding='same', activation='relu')(X_encode)

    pool_e = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                       padding="same")(conv_e)

    for i in range(1, len(filters)):
        conv_e = keras.layers.Conv2D(filters=filters[i],
                                     kernel_size=(3, 3),
                                     padding='same',
                                     activation='relu')(pool_e)
        pool_e = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           padding="same")(conv_e)

    latent_e = pool_e
    encoder = keras.Model(inputs=X_encode, outputs=latent_e)
    # encoder.summary()

    # ****************************** DECODER *********************************
    X_decode = keras.Input(shape=latent_dims)

    conv_d = keras.layers.Conv2D(filters=filters[-1], kernel_size=(3, 3),
                                 padding='same', activation='relu')(X_decode)

    pool_d = keras.layers.UpSampling2D((2, 2))(conv_d)

    for j in range(len(filters) - 2, 0, -1):
        conv_d = keras.layers.Conv2D(filters=filters[j], kernel_size=(3, 3),
                                     padding='same', activation='relu')(pool_d)
        pool_d = keras.layers.UpSampling2D((2, 2))(conv_d)

    conv_d = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                                 padding='valid', activation='relu')(pool_d)

    pool_d = keras.layers.UpSampling2D((2, 2))(conv_d)

    output = keras.layers.Conv2D(filters=input_dims[-1], kernel_size=(3, 3),
                                 padding='same', activation='sigmoid')(pool_d)

    decoder = keras.Model(inputs=X_decode, outputs=output)
    # decoder.summary()

    # ******************************* MODEL **********************************
    X_input = keras.Input(shape=input_dims)
    encode_output = encoder(X_input)
    decoder_output = decoder(encode_output)
    auto = keras.Model(inputs=X_input, outputs=decoder_output)
    # auto.summary()
    auto.compile(loss='binary_crossentropy', optimizer='adam')
    return (encoder, decoder, auto)
