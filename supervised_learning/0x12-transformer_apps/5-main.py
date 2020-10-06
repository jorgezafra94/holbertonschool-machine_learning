#!/usr/bin/env python3
import tensorflow.compat.v2 as tf 
import matplotlib.pyplot as plt
train_transformer = __import__('5-train').train_transformer
CustomSchedule= __import__('5-train').CustomSchedule
tf.compat.v1.enable_eager_execution()
transformer = train_transformer(6, 512, 8, 2048, 32, 40, 5)
