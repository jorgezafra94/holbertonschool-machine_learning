# Transfer Learning
Transfer Learning is a technique in Machine Learning, that consists in resolve a problem using a pre-trained-model created to solve a different problem.<br>
The idea is to use that knowledge (weights, bias, etc), and try to resolve a different problem; therefore we have to use different implementations to use that knowledge as an advantage to resolve the new problem.

### Task0 - Transfer Knowledge
Write a python script that trains a convolutional neural network to classify the CIFAR 10 dataset:<br>
<br>
* You must use one of the applications listed in Keras Applications
* Your script must save your trained model in the current working directory as cifar10.h5
* Your saved model should be compiled
* Your saved model should have a validation accuracy of 88% or higher
* Your script should not run when the file is imported
* Hint: The training may take a while, start early!
<br>
In the same file, write a function def preprocess_data(X, Y) that pre-processes the data for your model:<br>
<br>

*  X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data, where m is the number of data points
*  Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
*  Returns: X_p, Y_p
*  X_p is a numpy.ndarray containing the preprocessed X
*  Y_p is a numpy.ndarray containing the preprocessed Y

```
ubuntu-xenial:0x09-transfer_learning$ cat 0-main.py
#!/usr/bin/env python3

import tensorflow.keras as K
preprocess_data = __import__('0-transfer').preprocess_data

# fix issue with saving keras applications
K.learning_phase = K.backend.learning_phase 

_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10.h5')
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)
ubuntu-xenial:0x09-transfer_learning$ ./0-main.py
10000/10000 [==============================] - 159s 16ms/sample - loss: 0.3329 - acc: 0.8864
```
