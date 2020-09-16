#!/usr/bin/env python3
"""
forecasting Complete univariate and Multivariate
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
preprocess = __import__('preprocess_data').preprocess


# ************************************** UNIVARIATE FUNCTIONS ************
def create_time_steps(length):
    return list(range(-length, 0))


def univariate_data(dataset, start_index, end_index,
                    history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        # this create range of 24 from i to i + 24
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])

    return np.array(data), np.array(labels)


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(),
                     marker[i], label=labels[i])
        plt.legend()
        plt.xlim([time_steps[0], (future + 5) * 2])
        plt.xlabel('Time-Step')

    return plt


# ***************** Data Preprocessed ************************
data = preprocess()
data.reset_index(inplace=True)

# ****************** Data Loader Parameters ********************
BATCH_SIZE = 32
BUFFER_SIZE = 10000
TRAIN_SPLIT = int(len(data) - (24 * 2))

# LSTM Parameters
EVALUATION_INTERVAL = 200
EPOCHS = 20
PATIENCE = 5

# ********************** UNIVARIATE FORECAST ***************************
values_data = data['Close']
values_data.index = data['Timestamp']
# Normalization
values_mean = values_data[:int(TRAIN_SPLIT)].mean()
values_std = values_data[:int(TRAIN_SPLIT)].std()
uni_data = (values_data - values_mean) / values_std

univariate_past_history = 24
univariate_future_target = 1

x_t_uni, y_t_uni = univariate_data(dataset=uni_data,
                                   start_index=0,
                                   end_index=TRAIN_SPLIT,
                                   history_size=univariate_past_history,
                                   target_size=univariate_future_target)

x_val_uni, y_val_uni = univariate_data(dataset=uni_data,
                                       start_index=TRAIN_SPLIT,
                                       end_index=None,
                                       history_size=univariate_past_history,
                                       target_size=univariate_future_target)

train_univariate = tf.data.Dataset.from_tensor_slices((x_t_uni, y_t_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE)
train_univariate = train_univariate.batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(24, input_shape=x_t_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mse')

early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=3,
                                         restore_best_weights=True)
simple_lstm_model.fit(train_univariate,
                      epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate,
                      callbacks=[early],
                      validation_steps=50)
for x, y in val_univariate.take(3):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
                      simple_lstm_model.predict(x)[0]],
                     0, 'Simple LSTM model')
    plot.show()
