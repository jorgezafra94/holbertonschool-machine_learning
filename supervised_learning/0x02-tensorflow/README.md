# Tensorflow
Here we are going to learn how works this Framework of python
as we know the logic learned in DNN using perceptrons, we are going to try reply the Multiclass Classification using Tensorflow<br>
In orther to do that we have to follow the next tasks.<br>

## Task-0
Write the function `def create_placeholders(nx, classes):` that returns two placeholders, x and y, for the neural network:<br>

nx: the number of feature columns in our data<br>
classes: the number of classes in our classifier<br>
Returns: placeholders named x and y, respectively<br>
x is the placeholder for the input data to the neural network<br>
y is the placeholder for the one-hot labels for the input data<br>

we are going to use:

```
#!/usr/bin/env python3

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders

x, y = create_placeholders(784, 10)
print(x)
print(y)
```

the output should be

```
Tensor("x:0", shape=(?, 784), dtype=float32)
Tensor("y:0", shape=(?, 10), dtype=float32)
```


## Task-1
Write the function `def create_layer(prev, n, activation):`<br>

prev is the tensor output of the previous layer<br>
n is the number of nodes in the layer to create<br>
activation is the activation function that the layer should use<br>
use tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG") to implement He et. al initialization for the layer weights<br>
each layer should be given the name layer<br>
Returns: the tensor output of the layer<br>
we are going to use:

```
#!/usr/bin/env python3
import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
create_layer = __import__('1-create_layer').create_layer

x, y = create_placeholders(784, 10)
l = create_layer(x, 256, tf.nn.tanh)
print(l)
```

the output should be

```
Tensor("layer/Tanh:0", shape=(?, 256), dtype=float32)
```

## Task-2
Write the function `def forward_prop(x, layer_sizes=[], activations=[]):` that creates the forward propagation graph for the neural network:<br>

x is the placeholder for the input data<br>
layer_sizes is a list containing the number of nodes in each layer of the network<br>
activations is a list containing the activation functions for each layer of the network<br>
Returns: the prediction of the network in tensor form<br>
For this function, you should import your create_layer function with create_layer = __import__('1-create_layer').create_layer<br>
we are going to use:

```
#!/usr/bin/env python3
import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop

x, y = create_placeholders(784, 10)
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
print(y_pred)
```

the output should be

```
Tensor("layer_2/BiasAdd:0", shape=(?, 10), dtype=float32)
```

## Task-3
Write the function `def calculate_accuracy(y, y_pred):` that calculates the accuracy of a prediction:<br>

y is a placeholder for the labels of the input data<br>
y_pred is a tensor containing the network’s predictions<br>
Returns: a tensor containing the decimal accuracy of the prediction<br>
we are going to use:

```
#!/usr/bin/env python3

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy

x, y = create_placeholders(784, 10)
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
accuracy = calculate_accuracy(y, y_pred)
print(accuracy)
```

the output should be

```
Tensor("Mean:0", shape=(), dtype=float32)
```

## Task-4
Write the function `def calculate_loss(y, y_pred):` that calculates the softmax cross-entropy loss of a prediction:<br>

y is a placeholder for the labels of the input data<br>
y_pred is a tensor containing the network’s predictions<br>
Returns: a tensor containing the loss of the prediction<br>
we are going to use:

```
#!/usr/bin/env python3
import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_loss = __import__('4-calculate_loss').calculate_loss

x, y = create_placeholders(784, 10)
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
loss = calculate_loss(y, y_pred)
print(loss)
```

the output should be

```
Tensor("softmax_cross_entropy_loss/value:0", shape=(), dtype=float32)
```


## Task-5
Write the function `def create_train_op(loss, alpha):` that creates the training operation for the network:<br>

loss is the loss of the network’s prediction<br>
alpha is the learning rate<br>
Returns: an operation that trains the network using gradient descent<br>
we are going to use:

```
#!/usr/bin/env python3
import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


x, y = create_placeholders(784, 10)
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
loss = calculate_loss(y, y_pred)
train_op = create_train_op(loss, 0.01)
print(train_op)
```

the output should be

```
name: "GradientDescent"
op: "NoOp"
input: "^GradientDescent/update_layer/kernel/ApplyGradientDescent"
input: "^GradientDescent/update_layer/bias/ApplyGradientDescent"
input: "^GradientDescent/update_layer_1/kernel/ApplyGradientDescent"
input: "^GradientDescent/update_layer_1/bias/ApplyGradientDescent"
input: "^GradientDescent/update_layer_2/kernel/ApplyGradientDescent"
```

## Task-6
Write the function `def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):` that builds, trains, and saves a neural network classifier:<br>

X_train is a numpy.ndarray containing the training input data<br>
Y_train is a numpy.ndarray containing the training labels<br>
X_valid is a numpy.ndarray containing the validation input data<br>
Y_valid is a numpy.ndarray containing the validation labels<br>
layer_sizes is a list containing the number of nodes in each layer of the network<br>
actications is a list containing the activation functions for each layer of the network<br>
alpha is the learning rate<br>
iterations is the number of iterations to train over<br>
save_path designates where to save the model<br>
Add the following to the graph’s collection<br>
placeholders x and y<br>
tensors y_pred, loss, and accuracy<br>
operation train_op<br>
After every 100 iterations, the 0th iteration, and iterations iterations, print the following:<br>
After {i} iterations: where i is the iteration<br>
\tTraining Cost: {cost} where {cost} is the training cost<br>
\tTraining Accuracy: {accuracy} where {accuracy} is the training accuracy<br>
\tValidation Cost: {cost} where {cost} is the validation cost<br>
\tValidation Accuracy: {accuracy} where {accuracy} is the validation accuracy<br>
Reminder: the 0th iteration represents the model before any training has occurred<br>
After training has completed, save the model to save_path<br>
You may use the following imports:<br>
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy<br>
calculate_loss = __import__('4-calculate_loss').calculate_loss<br>
create_placeholders = __import__('0-create_placeholders').create_placeholders<br>
create_train_op = __import__('5-create_train_op').create_train_op<br>
forward_prop = __import__('2-forward_prop').forward_prop<br>
You are not allowed to use tf.saved_model<br>
Returns: the path where the model was saved<br>

we have to use:
```
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
train = __import__('6-train').train

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
    Y_train_oh = one_hot(Y_train, 10)
    X_valid_3D = lib['X_valid']
    Y_valid = lib['Y_valid']
    X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1))
    Y_valid_oh = one_hot(Y_valid, 10)

    layer_sizes = [256, 256, 10]
    activations = [tf.nn.tanh, tf.nn.tanh, None]
    alpha = 0.01
    iterations = 1000

    tf.set_random_seed(0)
    save_path = train(X_train, Y_train_oh, X_valid, Y_valid_oh, layer_sizes,
                      activations, alpha, iterations, save_path="./model.ckpt")
    print("Model saved in path: {}".format(save_path))
```

the output should be

```
2018-11-03 01:04:55.281078: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
After 0 iterations:
    Training Cost: 2.8232274055480957
    Training Accuracy: 0.08726000040769577
    Validation Cost: 2.810533285140991
    Validation Accuracy: 0.08640000224113464
After 100 iterations:
    Training Cost: 0.8393384218215942
    Training Accuracy: 0.7824000120162964
    Validation Cost: 0.7826032042503357
    Validation Accuracy: 0.8061000108718872
After 200 iterations:
    Training Cost: 0.6094841361045837
    Training Accuracy: 0.8396000266075134
    Validation Cost: 0.5562412142753601
    Validation Accuracy: 0.8597999811172485

...

After 1000 iterations:
    Training Cost: 0.352960467338562
    Training Accuracy: 0.9004999995231628
    Validation Cost: 0.32148978114128113
    Validation Accuracy: 0.909600019454956
Model saved in path: ./model.ckpt
```
## Task-7
Write the function `def evaluate(X, Y, save_path):` that evaluates the output of a neural network:

X is a numpy.ndarray containing the input data to evaluate
Y is a numpy.ndarray containing the one-hot labels for X
save_path is the location to load the model from
You are not allowed to use tf.saved_model
Returns: the network’s prediction, accuracy, and loss, respectively
we are going to use:

```
#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
evaluate = __import__('7-evaluate').evaluate

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_test_3D = lib['X_test']
    Y_test = lib['Y_test']
    X_test = X_test_3D.reshape((X_test_3D.shape[0], -1))
    Y_test_oh = one_hot(Y_test, 10)

    Y_pred_oh, accuracy, cost = evaluate(X_test, Y_test_oh, './model.ckpt')
    print("Test Accuracy:", accuracy)
    print("Test Cost:", cost)

    Y_pred = np.argmax(Y_pred_oh, axis=1)

    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_test_3D[i])
        plt.title(str(Y_test[i]) + ' : ' + str(Y_pred[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
```

the output should be the above one and a plot with the images

```
2018-11-03 02:08:30.767168: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
Test Accuracy: 0.9391
Test Cost: 0.21756475
```

