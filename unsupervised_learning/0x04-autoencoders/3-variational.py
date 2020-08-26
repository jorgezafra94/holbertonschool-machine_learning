#!/usr/bin/env python3
"""
Variational Autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    * input_dims is an integer containing the dimensions of the model input
    * hidden_layers is a list containing the number of nodes for each hidden
      layer in the encoder, respectively
    * the hidden layers should be reversed for the decoder
    * latent_dims is an integer containing the dimensions of the latent space
      representation

    Returns: encoder, decoder, auto
    * encoder is the encoder model, which should output the latent
      representation,
      the mean, and the log variance, respectively
    * decoder is the decoder model
    * auto is the full autoencoder model
    * The autoencoder model should be compiled using adam optimization and
      binary cross-entropy loss
    * All layers should use a relu activation except for the mean and log
      variance layers in the encoder, which should use None, and the last
      layer in the decoder, which should use sigmoid
    """
    #  ****************************  ENCODER ****************************
    X_encode = keras.Input(shape=(input_dims,))
    hidden_e = keras.layers.Dense(units=hidden_layers[0], activation='relu')
    Y_prev = hidden_e(X_encode)
    for i in range(1, len(hidden_layers)):
        hidden_e = keras.layers.Dense(units=hidden_layers[i],
                                      activation='relu')
        Y_prev = hidden_e(Y_prev)
    latent_lay = keras.layers.Dense(units=latent_dims, activation=None)
    z_mean = latent_lay(Y_prev)
    z_log_sigma = latent_lay(Y_prev)

    def sample_z(args):
        """
        Sampling function
        """
        mu, sigma = args
        batch = keras.backend.shape(mu)[0]
        dim = keras.backend.int_shape(mu)[1]
        eps = keras.backend.random_normal(shape=(batch, dim))
        return mu + keras.backend.exp(sigma / 2) * eps

    z = keras.layers.Lambda(sample_z,
                            output_shape=(latent_dims,))([z_mean,
                                                          z_log_sigma])
    encoder = keras.Model(inputs=X_encode,
                          outputs=[z, z_mean, z_log_sigma])
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
    # X_input = keras.Input(shape=(input_dims,))
    encode_output = encoder(X_encode)[-1]
    decoder_output = decoder(encode_output)
    auto = keras.Model(inputs=X_encode, outputs=decoder_output)
    # auto.summary()

    def vae_loss(x, x_decoded_mean):
        """
        loss function variational Autoencoder
        from keras documentation
        """
        xent_loss = keras.backend.binary_crossentropy(x, x_decoded_mean)
        xent_loss = keras.backend.sum(xent_loss, axis=1)
        first = 1 + z_log_sigma - keras.backend.square(z_mean)
        second = keras.backend.exp(z_log_sigma)
        kl_loss = -0.5 * keras.backend.mean(first - second, axis=-1)
        return xent_loss + kl_loss

    auto.compile(loss=vae_loss, optimizer='adam')
    return (encoder, decoder, auto)
