# Multiclass Classification
With this algorithm we are going to be able of classify diferent types of images <br>
the goal is to classify this pictures how you can see
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-multiclass_classification/data/multi-classification.png)
## Task 0 - Encode
### One-Hot-Encode
One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction.
<br>
<b> the One-hot-Encode will be our expected output </b>
this encode depends on:
* number of different classes
<br>
for example if we have a total of 10 different types of pictures each encode is going to have this numbers
now in this case we are going to use numerical classification so our encode depends of the value
if we have `Y = [5, 7, 8, 9, 0, 1, 2, 1]`
our encode should be like:

```
  5  7  8  9  0  1  2  1
[[0, 0, 0, 0, 1, 0, 1, 0]
 [0, 0, 0, 0, 0, 1, 0, 1]
 [0, 0, 0, 0, 0, 0, 0, 0]
 [0, 0, 0, 0, 0, 0, 0, 0]
 [0, 0, 0, 0, 0, 0, 0, 0]
 [1, 0, 0, 0, 0, 0, 0, 0]
 [0, 0, 0, 0, 0, 0, 0, 0]
 [0, 1, 0, 0, 0, 0, 0, 0]
 [0, 0, 1, 0, 0, 0, 0, 0]
 [0, 0, 0, 1, 0, 0, 0, 0]]
```
each column represent a value in Y
in this task we have to:
Write a function `def one_hot_encode(Y, classes):`that converts a numeric label vector into a one-hot matrix:
<br>
Y is a numpy.ndarray with shape (m,) containing numeric class labels
m is the number of examples
classes is the maximum number of classes found in Y
Returns: a one-hot encoding of Y with shape (classes, m), or None on failure
<br>
We are going to use:

```
#!/usr/bin/env python3

import numpy as np

oh_encode = __import__('0-one_hot_encode').one_hot_encode

lib = np.load('../data/MNIST.npz')
Y = lib['Y_train'][:10]

print(Y)
Y_one_hot = oh_encode(Y, 10)
print(Y_one_hot)
```

and we have to get

```
[5 0 4 1 9 2 1 3 1 4]
[[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 1. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]
```
## Task 1 - Decode
Write a function `def one_hot_decode(one_hot):` that converts a one-hot matrix into a vector of labels:

one_hot is a one-hot encoded numpy.ndarray with shape (classes, m)
classes is the maximum number of classes
m is the number of examples
Returns: a numpy.ndarray with shape (m, ) containing the numeric labels for each example, or None on failure
we have to use

```
#!/usr/bin/env python3

import numpy as np

oh_encode = __import__('0-one_hot_encode').one_hot_encode
oh_decode = __import__('1-one_hot_decode').one_hot_decode

lib = np.load('../data/MNIST.npz')
Y = lib['Y_train'][:10]

print(Y)
Y_one_hot = oh_encode(Y, 10)
Y_decoded = oh_decode(Y_one_hot)
print(Y_decoded)
```

we should get

```
[5 0 4 1 9 2 1 3 1 4]
[5 0 4 1 9 2 1 3 1 4]
```

## Task 2 - Pickle
### Pickle
It is a Module of Python that allows us to serialize an object and save it into a file
also we can load this object from the file

methods:
* dumps(object, file) --- store the object in a serialized way in the file
* load(object) ---- load the object from the open file
<br>
In this task we have to create the instance method `def save(self, filename):`

Saves the instance object to a file in pickle format
filename is the file to which the object should be saved
If filename does not have the extension .pkl, add it
<br>
Create the static method `def load(filename):`

Loads a pickled DeepNeuralNetwork object
filename is the file from which the object should be loaded
Returns: the loaded object, or None if filename doesn’t exist

we have to use:

```
#!/usr/bin/env python3

import numpy as np

Deep = __import__('2-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('0-one_hot_encode').one_hot_encode
one_hot_decode = __import__('1-one_hot_decode').one_hot_decode

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X_train.shape[0], [3, 1])
A, cost = deep.train(X_train, Y_train, iterations=500, graph=False)
deep.save('2-output')
del deep

saved = Deep.load('2-output.pkl')
A_saved, cost_saved = saved.evaluate(X_train, Y_train)

print(np.array_equal(A, A_saved) and cost == cost_saved)
```
we should get

```
Cost after 0 iterations: 0.7773240521521816
Cost after 100 iterations: 0.18751378071323066
Cost after 200 iterations: 0.12117095705345622
Cost after 300 iterations: 0.09031067302785326
Cost after 400 iterations: 0.07222364349190777
Cost after 500 iterations: 0.060335256947006956
True
ubuntu-xenial:0x01-multiclass_classification$ ls 2-output*
2-output.pkl
ubuntu-xenial:0x01-multiclass_classification$
```
## Task 3 - Update DNN to work as Multiclass classifier
### Softmax Function
Softmax Regression (synonyms: Multinomial Logistic, Maximum Entropy Classifier, or just Multi-class Logistic Regression) is a generalization of logistic regression that we can use for multi-class classification (under the assumption that the classes are mutually exclusive). In contrast, we use the (standard) Logistic Regression model in binary classification tasks.
<br>
This is the function that we <b>HAVE</b> to use in a Multiclass Classifier as activation function in the <b>last layer</b>

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/e348290cf48ddbb6e9a6ef4e39363568b67c09d3) <br>

In this task we have to update the class DeepNeuralNetwork to perform multiclass classification

You will need to update the instance methods forward_prop, cost, and evaluate
Y is now a one-hot numpy.ndarray of shape (classes, m)
Ideally, you should not have to change the __init__, gradient_descent, or train instance methods

Because the training process takes such a long time, I have pretrained a model for you to load and finish training ([3-saved.pkl](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-ml/3-saved.pkl)). This model has already been trained for 2000 iterations.

The training process may take up to 5 minutes

we have to use 

```
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Deep = __import__('3-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('0-one_hot_encode').one_hot_encode
one_hot_decode = __import__('1-one_hot_decode').one_hot_decode

lib= np.load('../data/MNIST.npz')
X_train_3D = lib['X_train']
Y_train = lib['Y_train']
X_valid_3D = lib['X_valid']
Y_valid = lib['Y_valid']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1)).T
Y_train_one_hot = one_hot_encode(Y_train, 10)
Y_valid_one_hot = one_hot_encode(Y_valid, 10)

deep = Deep.load('3-saved.pkl')
A_one_hot, cost = deep.train(X_train, Y_train_one_hot, iterations=100,
                             step=10, graph=False)
A = one_hot_decode(A_one_hot)
accuracy = np.sum(Y_train == A) / Y_train.shape[0] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))

A_one_hot, cost = deep.evaluate(X_valid, Y_valid_one_hot)
A = one_hot_decode(A_one_hot)
accuracy = np.sum(Y_valid == A) / Y_valid.shape[0] * 100
print("Validation cost:", cost)
print("Validation accuracy: {}%".format(accuracy))

deep.save('3-output')

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_valid_3D[i])
    plt.title(A[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

and we should get

```
Cost after 0 iterations: 0.4388904112857044
Cost after 10 iterations: 0.4377828804163359
Cost after 20 iterations: 0.43668839872612714
Cost after 30 iterations: 0.43560674736059446
Cost after 40 iterations: 0.43453771176806555
Cost after 50 iterations: 0.4334810815993252
Cost after 60 iterations: 0.43243665061046205
Cost after 70 iterations: 0.4314042165687683
Cost after 80 iterations: 0.4303835811615513
Cost after 90 iterations: 0.4293745499077264
Cost after 100 iterations: 0.42837693207206473
Train cost: 0.42837693207206473
Train accuracy: 88.442%
Validation cost: 0.39517557351173044
Validation accuracy: 89.64%
```

## Task 4 - Sigmoid and Tanh activation Functions
Here we have to adapt our code in order to classify images using sigmoid function or tanh function
<br>
In this taks we have to update the __init__ method to `def __init__(self, nx, layers, activation='sig'):`
activation represents the type of activation function used in the hidden layers
sig represents a sigmoid activation
tanh represents a tanh activation
if activation is not sig or tanh, raise a ValueError with the exception: activation must be 'sig' or 'tanh'
Create the private attribute `__activation` and set it to the value of activation
Create a getter for the private attribute `__activation`
Update the forward_prop and gradient_descent instance methods to use the `__activation` function in the hidden layers
Because the training process takes such a long time, I have pre-trained a model for you to load and finish training ([4-saved.pkl](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-ml/4-saved.pkl)). This model has already been trained for 2000 iterations.
<br>
we have to use:

```
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Deep3 = __import__('3-deep_neural_network').DeepNeuralNetwork
Deep4 = __import__('4-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('0-one_hot_encode').one_hot_encode
one_hot_decode = __import__('1-one_hot_decode').one_hot_decode

lib= np.load('../data/MNIST.npz')
X_train_3D = lib['X_train']
Y_train = lib['Y_train']
X_valid_3D = lib['X_valid']
Y_valid = lib['Y_valid']
X_test_3D = lib['X_test']
Y_test = lib['Y_test']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1)).T
X_test = X_test_3D.reshape((X_test_3D.shape[0], -1)).T
Y_train_one_hot = one_hot_encode(Y_train, 10)
Y_valid_one_hot = one_hot_encode(Y_valid, 10)
Y_test_one_hot = one_hot_encode(Y_test, 10)

print('Sigmoid activation:')
deep3 = Deep3.load('3-output.pkl')
A_one_hot3, cost3 = deep3.evaluate(X_train, Y_train_one_hot)
A3 = one_hot_decode(A_one_hot3)
accuracy3 = np.sum(Y_train == A3) / Y_train.shape[0] * 100
print("Train cost:", cost3)
print("Train accuracy: {}%".format(accuracy3))
A_one_hot3, cost3 = deep3.evaluate(X_valid, Y_valid_one_hot)
A3 = one_hot_decode(A_one_hot3)
accuracy3 = np.sum(Y_valid == A3) / Y_valid.shape[0] * 100
print("Validation cost:", cost3)
print("Validation accuracy: {}%".format(accuracy3))
A_one_hot3, cost3 = deep3.evaluate(X_test, Y_test_one_hot)
A3 = one_hot_decode(A_one_hot3)
accuracy3 = np.sum(Y_test == A3) / Y_test.shape[0] * 100
print("Test cost:", cost3)
print("Test accuracy: {}%".format(accuracy3))

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_test_3D[i])
    plt.title(A3[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

print('\nTanh activaiton:')

deep4 = Deep4.load('4-saved.pkl')
A_one_hot4, cost4 = deep4.train(X_train, Y_train_one_hot, iterations=100,
                                step=10, graph=False)
A4 = one_hot_decode(A_one_hot4)
accuracy4 = np.sum(Y_train == A4) / Y_train.shape[0] * 100
print("Train cost:", cost4)
print("Train accuracy: {}%".format(accuracy4))
A_one_hot4, cost4 = deep4.evaluate(X_valid, Y_valid_one_hot)
A4 = one_hot_decode(A_one_hot4)
accuracy4 = np.sum(Y_valid == A4) / Y_valid.shape[0] * 100
print("Validation cost:", cost4)
print("Validation accuracy: {}%".format(accuracy4))
A_one_hot4, cost4 = deep4.evaluate(X_test, Y_test_one_hot)
A4 = one_hot_decode(A_one_hot4)
accuracy4 = np.sum(Y_test == A4) / Y_test.shape[0] * 100
print("Test cost:", cost4)
print("Test accuracy: {}%".format(accuracy4))
deep4.save('4-output')

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_test_3D[i])
    plt.title(A4[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
```
we should get

```
Sigmoid activation:
Train cost: 0.42837693207206473
Train accuracy: 88.442%
Validation cost: 0.39517557351173044
Validation accuracy: 89.64%
Test cost: 0.4074169894615401
Test accuracy: 89.0%

Tanh activaiton:
Cost after 0 iterations: 0.18061815622291985
Cost after 10 iterations: 0.18012009542718577
Cost after 20 iterations: 0.1796242897834926
Cost after 30 iterations: 0.17913072860418564
Cost after 40 iterations: 0.1786394012066576
Cost after 50 iterations: 0.17815029691267442
Cost after 60 iterations: 0.1776634050478437
Cost after 70 iterations: 0.1771787149412177
Cost after 80 iterations: 0.1766962159250237
Cost after 90 iterations: 0.1762158973345138
Cost after 100 iterations: 0.1757377485079266
Train cost: 0.1757377485079266
Train accuracy: 95.006%
Validation cost: 0.17689309600397934
Validation accuracy: 95.13000000000001%
Test cost: 0.1809489808838737
Test accuracy: 94.77%
```
