# Data Augmentation

## Task0 - Flip
Write a function def flip_image(image): that flips an image horizontally:<br>
<br>
* image is a 3D tf.Tensor containing the image to flip

Returns the flipped image

```
$ cat 0-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
flip_image = __import__('0-flip').flip_image

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(0)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(flip_image(image))
    plt.show()
$ ./0-main.py
```
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/pipeline/0x03-data_augmentation/images/im1.PNG)

## Task1 - Crop 
Write a function def crop_image(image, size): that performs a random crop of an image:<br>
<br>
* image is a 3D tf.Tensor containing the image to crop
* size is a tuple containing the size of the crop

Returns the cropped image

```
$ cat 1-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
crop_image = __import__('1-crop').crop_image

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(1)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(crop_image(image, (200, 200, 3)))
    plt.show()
$ ./1-main.py
```
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/pipeline/0x03-data_augmentation/images/im2.PNG)

## Task2 - Rotate
Write a function def rotate_image(image): that rotates an image by 90 degrees counter-clockwise:<br>
<br>
* image is a 3D tf.Tensor containing the image to rotate

Returns the rotated image

```
$ cat 2-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
rotate_image = __import__('2-rotate').rotate_image

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(2)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(rotate_image(image))
    plt.show()
$ ./2-main.py
```
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/pipeline/0x03-data_augmentation/images/im3.PNG)

## Task3 - Shear
Write a function def shear_image(image, intensity): that randomly shears an image:<br>
<br>
* image is a 3D tf.Tensor containing the image to shear
* intensity is the intensity with which the image should be sheared

Returns the sheared image

```
$ cat 3-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
shear_image = __import__('3-shear').shear_image

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(3)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(shear_image(image, 50))
    plt.show()
$ ./3-main.py
```
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/pipeline/0x03-data_augmentation/images/im4.PNG)

## Task4 - Brightness
Write a function def change_brightness(image, max_delta): that randomly changes the brightness of an image:<br>
<br>
* image is a 3D tf.Tensor containing the image to change
* max_delta is the maximum amount the image should be brightened (or darkened)

Returns the altered image

```
$ cat 4-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
change_brightness = __import__('4-brightness').change_brightness

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(4)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(change_brightness(image, 0.3))
    plt.show()
$ ./4-main.py
```
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/pipeline/0x03-data_augmentation/images/im5.PNG)

## Task5 - Hue
Write a function def change_hue(image, delta): that changes the hue of an image:<br>
<br>
* image is a 3D tf.Tensor containing the image to change
* delta is the amount the hue should change

Returns the altered image

```
$ cat 5-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
change_hue = __import__('5-hue').change_hue

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(5)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(change_hue(image, -0.5))
    plt.show()
$ ./5-main.py
```

![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/pipeline/0x03-data_augmentation/images/im6.PNG)

## Task100 - PCA
Write a function def pca_color(image, alphas): that performs PCA color augmentation as described in the AlexNet paper:<br>
<br>
* image is a 3D tf.Tensor containing the image to change
* alphas a tuple of length 3 containing the amount that each channel should change

Returns the augmented image

```
$ cat 100-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
pca_color = __import__('100-pca').pca_color

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(100)
np.random.seed(100)
doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    alphas = np.random.normal(0, 0.1, 3)
    plt.imshow(pca_color(image, alphas))
    plt.show()
$ ./100-main.py
```
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/pipeline/0x03-data_augmentation/images/im7.PNG)
