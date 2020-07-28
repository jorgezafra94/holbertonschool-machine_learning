# Supervised Learning
Here are most important Methods of Supervised Training in Machine Learning Nowadays
## Binary Classification
This algorithm is able to Say if a picture is or not a specific class, I mean for example if a picture is a dog or it isnt.
Here we are going to learn concepts as:
* Perceptron
* Sigmoid Activation Function
* Lost Function (logarithmic logistic)
* Cost
* Neural Network
* Deep Neural Network
## Multiclass Classification
here we are going to use sigmoid and tanh activation functions <br>
also we are going to learn:
* One-Hot-Encode
* pickle Module in Python in order to store objects
* Multiple classification using DNN
## Tensorflow
here we are going to reply de Multiclass Classifier using tensorflow<br>
## Optimization
here we are going to use optimizations for training like:
* mini_batch training
* learning_rate_decay (alpha hyperparameter)
* Adam, gradient descent with momentum and RMSprop
* batch_normalization
## Error Analisys
here we are going to learn:
* what to do with high bias problem 
* what to do with high variance problem
* confusion matrix
* sensitivity, specificity, f1-score, TN, TP, FP, FN
## Regularization
methods that you can apply in your NN to prevent overfitting
* L2 & L1
* Dropout
* Data Augmentation
* Early Stopping
## Keras
Start to learn Keras and how to use it
* create model using Sequential method and Input methos
* config model using compile
* train model
* evaluate or test the model
* predict with the model created
## CNN
Here we are going to learn how to play with convolutional Neural Networks using filters and max, avg pooling
* forward conv layer
* forward pool layer
* backprop conv layer
* backprop pool layer
* creating LeNet5 Model in Tensorflow as in Keras
## deep_CNN
Here we are going to create different arquitectures that were revolutionary in their time
* Inception Network
* ResNet Network
* DenseNet Network
## Transfer Learning
Here we are going to use transfer Learning technique to create a model from one of the models in application keras that perform an val_Acc >= 0.88, methods that i used to obtain that:
* Using prediction as input
* freezing the whole pre_model
* training the whole pre_model
* freezing half and training half of pre_model
## Object Detection
Here we are going to create ou own Yolo3 using darknet to process the images
* processing outputs of Darkent
* filter by threshold
* filter using IoU
* Load images (database)
* preprocess the images for the Darknet
* detecting objects
* opencv
## Face verification
* using yolo logic to predict
* triplet loss
* creating a model including layers in keras
* evaluate distance between pictures after embeding
* use opencv cv2
