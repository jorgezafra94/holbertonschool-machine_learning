#!/usr/bin/env python3
"""
Transfer Learning
"""

import tensorflow.keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
    * X is a numpy.ndarray of shape (m, 32, 32, 3) containing
        the CIFAR 10 data, where m is the number of data points
    * Y is a numpy.ndarray of shape (m,) containing the CIFAR 10
        labels for X
        Returns: X_p, Y_p
    * X_p is a numpy.ndarray containing the preprocessed X
    * Y_p is a numpy.ndarray containing the preprocessed Y
    """

    entrada = K.Input(shape=(32, 32, 3))
    resize = K.layers.Lambda(lambda image:
                             tf.image.resize(image, (150, 150)))(entrada)
    dense169 = K.applications.DenseNet169(include_top=False,
                                          weights="imagenet",
                                          input_tensor=resize)
    out = dense169(resize)
    pre_model = K.models.Model(inputs=entrada, outputs=out)

    X_p = K.applications.densenet.preprocess_input(X)
    features = pre_model.predict(X_p)
    Y_p = K.utils.to_categorical(y=Y, num_classes=10)
    return (features, Y_p)


if __name__ == "__main__":
    # getting info
    Train, Test = K.datasets.cifar10.load_data()
    (x_train, y_train) = Train
    (x_test, y_test) = Test

    # preprocessing
    x_p, y_p = preprocess_data(x_train, y_train)
    X_p, Y_p = preprocess_data(x_test, y_test)

    lay_init = K.initializers.he_normal()

    # our model
    new_input = K.Input(shape=x_p.shape[1:])
    vector = K.layers.Flatten()(new_input)

    drop1 = K.layers.Dropout(0.3)(vector)
    norm_lay1 = K.layers.BatchNormalization()(drop1)
    FC1 = K.layers.Dense(units=510, activation='relu',
                         kernel_initializer=lay_init)(norm_lay1)
    norm_lay2 = K.layers.BatchNormalization()(FC1)
    out = K.layers.Dense(units=10, activation='softmax',
                         kernel_initializer=lay_init)(norm_lay2)

    model = K.models.Model(inputs=new_input, outputs=out)

    learn_dec = K.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                              factor=0.1, patience=2)

    early = K.callbacks.EarlyStopping(patience=5)
    save = K.callbacks.ModelCheckpoint(filepath='cifar10.h5',
                                       save_best_only=True,
                                       monitor='val_acc',
                                       mode='max')

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x=x_p, y=y_p, batch_size=32, epochs=15,
              verbose=1, validation_data=(X_p, Y_p),
              callbacks=[save, early, learn_dec])
