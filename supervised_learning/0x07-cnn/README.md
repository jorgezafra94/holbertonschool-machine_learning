# Convolutional Neural Networks
## Task0 - Convolutional Forward Prop 
Write a function `def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):` that performs forward propagation over a convolutional layer of a neural network:
<br>
* A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the output of the previous layer<br>
* * m is the number of examples<br>
* * h_prev is the height of the previous layer<br>
* * w_prev is the width of the previous layer<br>
* * c_prev is the number of channels in the previous layer<br>
* W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels for the convolution<br>
* * kh is the filter height<br>
* * kw is the filter width<br>
* * c_prev is the number of channels in the previous layer<br>
* * c_new is the number of channels in the output<br>
* b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied to the convolution<br>
* activation is an activation function applied to the convolution<br>
* padding is a string that is either same or valid, indicating the type of padding used<br>
* stride is a tuple of (sh, sw) containing the strides for the convolution<br>
* * sh is the stride for the height<br>
* * sw is the stride for the width<br>
you may import numpy as np<br>
Returns: the output of the convolutional layer<br>
use this main file
```
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
conv_forward = __import__('0-conv_forward').conv_forward

if __name__ == "__main__":
    np.random.seed(0)
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    m, h, w = X_train.shape
    X_train_c = X_train.reshape((-1, h, w, 1))

    W = np.random.randn(3, 3, 1, 2)
    b = np.random.randn(1, 1, 1, 2)

    def relu(Z):
        return np.maximum(Z, 0)

    plt.imshow(X_train[0])
    plt.show()
    A = conv_forward(X_train_c, W, b, relu, padding='valid')
    print(A.shape)
    plt.imshow(A[0, :, :, 0])
    plt.show()
    plt.imshow(A[0, :, :, 1])
    plt.show()
```
the output should be<br>
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/pictures/input-0.png)<br>
`(50000, 26, 26, 2)`<br>
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/pictures/output-0.1.png)
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/pictures/output-0.2.png)<br>
## Task1 - Pooling Forward Prop
Write a function `def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):` that performs forward propagation over a pooling layer of a neural network:
<br>
* A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the output of the previous layer<br>
* * m is the number of examples<br>
* * h_prev is the height of the previous layer<br>
* * w_prev is the width of the previous layer<br>
* * c_prev is the number of channels in the previous layer<br>
* kernel_shape is a tuple of (kh, kw) containing the size of the kernel for the pooling<br>
* * kh is the kernel height<br>
* * kw is the kernel width<br>
* stride is a tuple of (sh, sw) containing the strides for the pooling<br>
* * sh is the stride for the height<br>
* * sw is the stride for the width<br>
* mode is a string containing either max or avg, indicating whether to perform maximum or average pooling, respectively<br>
you may import numpy as np<br>
Returns: the output of the pooling layer<br>
use this main

```
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
pool_forward = __import__('1-pool_forward').pool_forward

if __name__ == "__main__":
    np.random.seed(0)
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    m, h, w = X_train.shape
    X_train_a = X_train.reshape((-1, h, w, 1))
    X_train_b = 1 - X_train_a
    X_train_c = np.concatenate((X_train_a, X_train_b), axis=3)

    print(X_train_c.shape)
    plt.imshow(X_train_c[0, :, :, 0])
    plt.show()
    plt.imshow(X_train_c[0, :, :, 1])
    plt.show()
    A = pool_forward(X_train_c, (2, 2), stride=(2, 2))
    print(A.shape)
    plt.imshow(A[0, :, :, 0])
    plt.show()
    plt.imshow(A[0, :, :, 1])
    plt.show()
```
the output should be like<br>
`(50000, 28, 28, 2)`<br>
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/pictures/input-1.png)<br>
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/pictures/input-1.2.png)<br>
`(50000, 14, 14, 2)`<br>
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/pictures/output-1.1.png)<br>
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/pictures/output-1.2.png)
<br>
## Task2 - Convolutional Back Prop
Write a function `def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):` that performs back propagation over a convolutional layer of a neural network:
<br>
* dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the partial derivatives with respect to the unactivated output of the convolutional layer<br>
* * m is the number of examples
* * h_new is the height of the output
* * w_new is the width of the output
* * c_new is the number of channels in the output
* A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the output of the previous layer
* * h_prev is the height of the previous layer
* * w_prev is the width of the previous layer
* * c_prev is the number of channels in the previous layer
* W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels for the convolution
* * kh is the filter height
* * kw is the filter width
* b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied to the convolution
* padding is a string that is either same or valid, indicating the type of padding used
* stride is a tuple of (sh, sw) containing the strides for the convolution
* * sh is the stride for the height
* * sw is the stride for the width<br>
you may import numpy as np<br>
Returns: the partial derivatives with respect to the previous layer (dA_prev), the kernels (dW), and the biases (db), respectively<br>
use this main<br>
```
#!/usr/bin/env python3

import numpy as np
conv_backward = __import__('2-conv_backward').conv_backward

if __name__ == "__main__":
    np.random.seed(0)
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    _, h, w = X_train.shape
    X_train_c = X_train[:10].reshape((-1, h, w, 1))

    W = np.random.randn(3, 3, 1, 2)
    b = np.random.randn(1, 1, 1, 2)

    dZ = np.random.randn(10, h - 2, w - 2, 2)
    print(conv_backward(dZ, X_train_c, W, b, padding="valid"))
```
the output should be like
```
(array([[[[-4.24205748],
         [ 0.19390938],
         [-2.80168847],
         ...,
         [-2.93059274],
         [-0.74257184],
         [ 1.23556676]],

        [[-1.00865794],
         [ 0.24929631],
         [ 2.99153975],
         ...,
         [ 0.43357888],
         [ 4.96415936],
         [-0.44667327]],

        [[-1.87909273],
         [-1.52898354],
         [-1.03988664],
         ...,
         [-2.08719794],
         [ 0.72118428],
         [ 0.43712847]],

        ...,

        [[-1.85981381],
         [-4.35543293],
         [ 3.57636107],
         ...,
         [ 2.11136296],
         [ 0.53958723],
         [-3.52000282]],

        [[-1.0499573 ],
         [-2.04696766],
         [-3.65137871],
         ...,
         [-0.52756967],
         [-0.08825488],
         [ 0.62158883]],

        [[-0.33383597],
         [-0.68426308],
         [-1.16737412],
         ...,
         [ 0.38028383],
         [ 0.90910959],
         [ 1.1018034 ]]],


       [[[ 1.3242862 ],
         [ 3.35050521],
         [-2.61244835],
         ...,
         [-5.35657632],
         [ 0.76179689],
         [ 2.18585273]],

        [[ 0.41947984],
         [ 2.29805997],
         [ 0.70448521],
         ...,
         [-0.15055621],
         [-1.85010471],
         [ 0.22182008]],

        [[-0.44134373],
         [ 1.70998625],
         [-0.43519259],
         ...,
         [-0.84228164],
         [ 0.06743225],
         [-0.33952493]],

        ...,

        [[-0.84072841],
         [ 2.23096657],
         [ 4.2740757 ],
         ...,
         [-1.64328314],
         [-1.00825088],
         [ 0.06493264]],

        [[ 0.51461905],
         [ 1.74947396],
         [ 3.12442805],
         ...,
         [ 2.64632352],
         [ 1.11166957],
         [-2.17004665]],

        [[-0.15977939],
         [ 0.71088702],
         [ 0.58870058],
         ...,
         [ 0.79037467],
         [-1.872449  ],
         [ 0.22958953]]],


       [[[-2.55102529],
         [-1.43443829],
         [-6.43429192],
         ...,
         [ 4.43919873],
         [-2.3961974 ],
         [ 1.12105391]],

        [[-3.49933601],
         [ 2.97808   ],
         [-5.94765644],
         ...,
         [-1.52227952],
         [ 0.71633969],
         [-2.69268038]],

        [[-0.6049378 ],
         [ 3.00515277],
         [-3.82581326],
         ...,
         [-0.82612782],
         [ 1.10270878],
         [ 0.57341665]],

        ...,

        [[ 2.47206612],
         [ 6.12030267],
         [ 4.85570283],
         ...,
         [ 1.7069348 ],
         [-3.26558701],
         [-2.19265787]],

        [[ 0.82794065],
         [ 2.50876332],
         [ 4.94170337],
         ...,
         [-4.11611469],
         [-1.89129533],
         [ 1.02817795]],

        [[ 0.61583613],
         [ 1.21100799],
         [ 1.26340831],
         ...,
         [-1.46870175],
         [-2.48288945],
         [-2.49803816]]],


       ...,


       [[[ 0.36480084],
         [ 4.05009666],
         [ 2.40882213],
         ...,
         [-1.39742733],
         [-1.58184928],
         [ 1.5492834 ]],

        [[-0.59246796],
         [-5.14195445],
         [-4.73361645],
         ...,
         [-6.29937402],
         [ 2.57781547],
         [-6.22413954]],

        [[ 0.28940123],
         [ 3.30399397],
         [-9.92107171],
         ...,
         [-4.7873951 ],
         [-5.51345667],
         [ 2.59603062]],

        ...,

        [[ 0.31895703],
         [ 2.7620854 ],
         [ 2.40446498],
         ...,
         [ 2.68160757],
         [ 2.3774331 ],
         [-5.17924359]],

        [[-0.84079478],
         [ 0.92656007],
         [ 1.69220611],
         ...,
         [ 0.23381858],
         [ 0.65019692],
         [ 2.52647242]],

        [[-0.21035363],
         [-0.49657321],
         [-0.97588817],
         ...,
         [ 1.37568796],
         [ 0.75783393],
         [-2.06076966]]],


       [[[-0.8764177 ],
         [ 0.04226753],
         [-3.92342249],
         ...,
         [-3.04784534],
         [-0.40436888],
         [ 0.42939003]],

        [[-1.99854061],
         [-1.36763433],
         [-3.31601105],
         ...,
         [ 3.56163624],
         [-5.45977866],
         [-1.1221114 ]],

        [[-2.97880521],
         [-7.02474334],
         [-2.6208715 ],
         ...,
         [-2.66868613],
         [-3.35947227],
         [ 1.52739149]],

        ...,

        [[-0.76204177],
         [-2.39471119],
         [ 1.88614862],
         ...,
         [ 8.52140674],
         [ 2.87244213],
         [ 5.4831909 ]],

        [[-0.28094631],
         [-1.54524622],
         [-2.26649997],
         ...,
         [ 4.01337541],
         [ 1.72949251],
         [ 0.26894907]],

        [[ 0.2333244 ],
         [ 0.15360826],
         [ 0.61304729],
         ...,
         [ 4.82873779],
         [ 1.58564885],
         [ 3.77278834]]],


       [[[-4.64117569],
         [-6.30127451],
         [-1.35549413],
         ...,
         [ 5.73490276],
         [ 4.48763997],
         [ 0.90584946]],

        [[-2.06780074],
         [ 0.74310235],
         [ 2.32306348],
         ...,
         [-1.93057052],
         [ 1.73865934],
         [ 1.29870813]],

        [[ 0.48429556],
         [-3.18452582],
         [-3.1882709 ],
         ...,
         [ 1.14229413],
         [-0.68614631],
         [ 0.48510011]],

        ...,

        [[ 1.31359094],
         [ 1.80393793],
         [-2.56324511],
         ...,
         [ 1.87402318],
         [ 2.10343171],
         [ 4.90609163]],

        [[ 0.984754  ],
         [ 0.49587505],
         [-0.26741779],
         ...,
         [ 1.93306272],
         [ 3.19125427],
         [-0.9173847 ]],

        [[ 0.87318188],
         [ 0.96086254],
         [ 1.69739496],
         ...,
         [-0.28586324],
         [ 2.24643738],
         [ 0.74045003]]]]), array([[[[ 10.13352674, -25.15674655]],

        [[ 33.27872337, -64.99062958]],

        [[ 31.29539025, -77.29275492]]],


       [[[ 10.61025981, -31.7337223 ]],

        [[ 10.34048231, -65.19271124]],

        [[ -1.73024336, -76.98703808]]],


       [[[ -1.49204439, -33.46094911]],

        [[  4.04542976, -63.47295685]],

        [[  2.9243666 , -64.29296016]]]]), array([[[[-113.18404846, -121.902714  ]]]]))
```
## Task3 - Pooling Back Prop
Write a function `def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):` that performs back propagation over a pooling layer of a neural network:
<br>
* dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the partial derivatives with respect to the output of the pooling layer
* * m is the number of examples
* * h_new is the height of the output
* * w_new is the width of the output
* * c is the number of channels
* A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing the output of the previous layer
* * h_prev is the height of the previous layer
* * w_prev is the width of the previous layer
* kernel_shape is a tuple of (kh, kw) containing the size of the kernel for the pooling
* * kh is the kernel height
* * kw is the kernel width
* stride is a tuple of (sh, sw) containing the strides for the pooling
* * sh is the stride for the height
* * sw is the stride for the width
* mode is a string containing either max or avg, indicating whether to perform maximum or average pooling, respectively<br>
you may import numpy as np<br>
Returns: the partial derivatives with respect to the previous layer (dA_prev)<br>
use this main
```
#!/usr/bin/env python3

import numpy as np
pool_backward = __import__('3-pool_backward').pool_backward

if __name__ == "__main__":
    np.random.seed(0)
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    _, h, w = X_train.shape
    X_train_a = X_train[:10].reshape((-1, h, w, 1))
    X_train_b = 1 - X_train_a
    X_train_c = np.concatenate((X_train_a, X_train_b), axis=3)

    dA = np.random.randn(10, h // 3, w // 3, 2)
    print(pool_backward(dA, X_train_c, (3, 3), stride=(3, 3)))
```
the output should be like
```
[[[[ 1.76405235  0.40015721]
   [ 1.76405235  0.40015721]
   [ 1.76405235  0.40015721]
   ...
   [ 1.49407907 -0.20515826]
   [ 1.49407907 -0.20515826]
   [ 0.          0.        ]]

  [[ 1.76405235  0.40015721]
   [ 1.76405235  0.40015721]
   [ 1.76405235  0.40015721]
   ...
   [ 1.49407907 -0.20515826]
   [ 1.49407907 -0.20515826]
   [ 0.          0.        ]]

  [[ 1.76405235  0.40015721]
   [ 1.76405235  0.40015721]
   [ 1.76405235  0.40015721]
   ...
   [ 1.49407907 -0.20515826]
   [ 1.49407907 -0.20515826]
   [ 0.          0.        ]]

  ...

  [[ 2.38314477  0.94447949]
   [ 2.38314477  0.94447949]
   [ 2.38314477  0.94447949]
   ...
   [-0.49803245  1.92953205]
   [-0.49803245  1.92953205]
   [ 0.          0.        ]]

  [[ 2.38314477  0.94447949]
   [ 2.38314477  0.94447949]
   [ 2.38314477  0.94447949]
   ...
   [-0.49803245  1.92953205]
   [-0.49803245  1.92953205]
   [ 0.          0.        ]]

  [[ 0.          0.        ]
   [ 0.          0.        ]
   [ 0.          0.        ]
   ...
   [ 0.          0.        ]
   [ 0.          0.        ]
   [ 0.          0.        ]]]


 [[[ 0.94942081  0.08755124]
   [ 0.94942081  0.08755124]
   [ 0.94942081  0.08755124]
   ...
   [-0.4555325   0.01747916]
   [-0.4555325   0.01747916]
   [ 0.          0.        ]]

  [[ 0.94942081  0.08755124]
   [ 0.94942081  0.08755124]
   [ 0.94942081  0.08755124]
   ...
   [-0.4555325   0.01747916]
   [-0.4555325   0.01747916]
   [ 0.          0.        ]]

  [[ 0.94942081  0.08755124]
   [ 0.94942081  0.08755124]
   [ 0.94942081  0.08755124]
   ...
   [-0.4555325   0.01747916]
   [-0.4555325   0.01747916]
   [ 0.          0.        ]]

  ...

  [[-1.33425847 -1.34671751]
   [-1.33425847 -1.34671751]
   [-1.33425847 -1.34671751]
   ...
   [ 0.27451636 -0.89091508]
   [ 0.27451636 -0.89091508]
   [ 0.          0.        ]]

  [[-1.33425847 -1.34671751]
   [-1.33425847 -1.34671751]
   [-1.33425847 -1.34671751]
   ...
   [ 0.27451636 -0.89091508]
   [ 0.27451636 -0.89091508]
   [ 0.          0.        ]]

  [[ 0.          0.        ]
   [ 0.          0.        ]
   [ 0.          0.        ]
   ...
   [ 0.          0.        ]
   [ 0.          0.        ]
   [ 0.          0.        ]]]


 [[[-1.15735526 -0.31229225]
   [-1.15735526 -0.31229225]
   [-1.15735526 -0.31229225]
   ...
   [ 1.0685094  -0.4533858 ]
   [ 1.0685094  -0.4533858 ]
   [ 0.          0.        ]]

  [[-1.15735526 -0.31229225]
   [-1.15735526 -0.31229225]
   [-1.15735526 -0.31229225]
   ...
   [ 1.0685094  -0.4533858 ]
   [ 1.0685094  -0.4533858 ]
   [ 0.          0.        ]]

  [[-1.15735526 -0.31229225]
   [-1.15735526 -0.31229225]
   [-1.15735526 -0.31229225]
   ...
   [ 1.0685094  -0.4533858 ]
   [ 1.0685094  -0.4533858 ]
   [ 0.          0.        ]]

  ...

  [[-2.28862004  0.25148442]
   [-2.28862004  0.25148442]
   [-2.28862004  0.25148442]
   ...
   [ 0.86259601 -2.65561909]
   [ 0.86259601 -2.65561909]
   [ 0.          0.        ]]

  [[-2.28862004  0.25148442]
   [-2.28862004  0.25148442]
   [-2.28862004  0.25148442]
   ...
   [ 0.86259601 -2.65561909]
   [ 0.86259601 -2.65561909]
   [ 0.          0.        ]]

  [[ 0.          0.        ]
   [ 0.          0.        ]
   [ 0.          0.        ]
   ...
   [ 0.          0.        ]
   [ 0.          0.        ]
   [ 0.          0.        ]]]


 ...


 [[[-0.12824197 -0.27177157]
   [-0.12824197 -0.27177157]
   [-0.12824197 -0.27177157]
   ...
   [-0.03331928 -1.5308035 ]
   [-0.03331928 -1.5308035 ]
   [ 0.          0.        ]]

  [[-0.12824197 -0.27177157]
   [-0.12824197 -0.27177157]
   [-0.12824197 -0.27177157]
   ...
   [-0.03331928 -1.5308035 ]
   [-0.03331928 -1.5308035 ]
   [ 0.          0.        ]]

  [[-0.12824197 -0.27177157]
   [-0.12824197 -0.27177157]
   [-0.12824197 -0.27177157]
   ...
   [-0.03331928 -1.5308035 ]
   [-0.03331928 -1.5308035 ]
   [ 0.          0.        ]]

  ...

  [[-1.31771734  0.54100822]
   [-1.31771734  0.54100822]
   [-1.31771734  0.54100822]
   ...
   [ 1.60852216 -0.56157875]
   [ 1.60852216 -0.56157875]
   [ 0.          0.        ]]

  [[-1.31771734  0.54100822]
   [-1.31771734  0.54100822]
   [-1.31771734  0.54100822]
   ...
   [ 1.60852216 -0.56157875]
   [ 1.60852216 -0.56157875]
   [ 0.          0.        ]]

  [[ 0.          0.        ]
   [ 0.          0.        ]
   [ 0.          0.        ]
   ...
   [ 0.          0.        ]
   [ 0.          0.        ]
   [ 0.          0.        ]]]


 [[[ 0.20727075  0.30773257]
   [ 0.20727075  0.30773257]
   [ 0.20727075  0.30773257]
   ...
   [-0.25595767 -0.34804638]
   [-0.25595767 -0.34804638]
   [ 0.          0.        ]]

  [[ 0.20727075  0.30773257]
   [ 0.20727075  0.30773257]
   [ 0.20727075  0.30773257]
   ...
   [-0.25595767 -0.34804638]
   [-0.25595767 -0.34804638]
   [ 0.          0.        ]]

  [[ 0.20727075  0.30773257]
   [ 0.20727075  0.30773257]
   [ 0.20727075  0.30773257]
   ...
   [-0.25595767 -0.34804638]
   [-0.25595767 -0.34804638]
   [ 0.          0.        ]]

  ...

  [[ 0.1474344  -0.97746488]
   [ 0.1474344  -0.97746488]
   [ 0.1474344  -0.97746488]
   ...
   [-1.00432271 -0.99819173]
   [-1.00432271 -0.99819173]
   [ 0.          0.        ]]

  [[ 0.1474344  -0.97746488]
   [ 0.1474344  -0.97746488]
   [ 0.1474344  -0.97746488]
   ...
   [-1.00432271 -0.99819173]
   [-1.00432271 -0.99819173]
   [ 0.          0.        ]]

  [[ 0.          0.        ]
   [ 0.          0.        ]
   [ 0.          0.        ]
   ...
   [ 0.          0.        ]
   [ 0.          0.        ]
   [ 0.          0.        ]]]


 [[[-1.37304255 -1.06774201]
   [-1.37304255 -1.06774201]
   [-1.37304255 -1.06774201]
   ...
   [-0.02852887 -0.32473027]
   [-0.02852887 -0.32473027]
   [ 0.          0.        ]]

  [[-1.37304255 -1.06774201]
   [-1.37304255 -1.06774201]
   [-1.37304255 -1.06774201]
   ...
   [-0.02852887 -0.32473027]
   [-0.02852887 -0.32473027]
   [ 0.          0.        ]]

  [[-1.37304255 -1.06774201]
   [-1.37304255 -1.06774201]
   [-1.37304255 -1.06774201]
   ...
   [-0.02852887 -0.32473027]
   [-0.02852887 -0.32473027]
   [ 0.          0.        ]]

  ...

  [[-1.00603686 -0.82154983]
   [-1.00603686 -0.82154983]
   [-1.00603686 -0.82154983]
   ...
   [ 1.00859729  0.52759738]
   [ 1.00859729  0.52759738]
   [ 0.          0.        ]]

  [[-1.00603686 -0.82154983]
   [-1.00603686 -0.82154983]
   [-1.00603686 -0.82154983]
   ...
   [ 1.00859729  0.52759738]
   [ 1.00859729  0.52759738]
   [ 0.          0.        ]]

  [[ 0.          0.        ]
   [ 0.          0.        ]
   [ 0.          0.        ]
   ...
   [ 0.          0.        ]
   [ 0.          0.        ]
   [ 0.          0.        ]]]]
```
## Task4 - LeNet-5 (Tensorflow) 
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/pictures/letnet%20config.png)<br>
Write a function `def lenet5(x, y):` that builds a modified version of the LeNet-5 architecture using tensorflow:

* x is a tf.placeholder of shape (m, 28, 28, 1) containing the input images for the network
* * m is the number of images
* y is a tf.placeholder of shape (m, 10) containing the one-hot labels for the network
* The model should consist of the following layers in order:
* Convolutional layer with 6 kernels of shape 5x5 with same padding
* Max pooling layer with kernels of shape 2x2 with 2x2 strides
* Convolutional layer with 16 kernels of shape 5x5 with valid padding
* Max pooling layer with kernels of shape 2x2 with 2x2 strides
* Fully connected layer with 120 nodes
* Fully connected layer with 84 nodes
* Fully connected softmax output layer with 10 nodes
* All layers requiring initialization should initialize their kernels with the he_normal initialization method: tf.contrib.layers.variance_scaling_initializer()
* All hidden layers requiring activation should use the relu activation function
* you may import tensorflow as tf
* you may NOT use tf.keras
* Returns:
* * a tensor for the softmax activated output
* * a training operation that utilizes Adam optimization (with default hyperparameters)
* * a tensor for the loss of the netowrk
* * a tensor for the accuracy of the network
use this main
```
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
lenet5 = __import__('4-lenet5').lenet5

if __name__ == "__main__":
    np.random.seed(0)
    tf.set_random_seed(0)
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    Y_train = lib['Y_train']
    X_valid = lib['X_valid']
    Y_valid = lib['Y_valid']
    m, h, w = X_train.shape
    X_train_c = X_train.reshape((-1, h, w, 1))
    X_valid_c = X_valid.reshape((-1, h, w, 1))
    x = tf.placeholder(tf.float32, (None, h, w, 1))
    y = tf.placeholder(tf.int32, (None,))
    y_oh = tf.one_hot(y, 10)
    y_pred, train_op, loss, acc = lenet5(x, y_oh)
    batch_size = 32
    epochs = 10
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            cost, accuracy = sess.run((loss, acc), feed_dict={x:X_train_c, y:Y_train})
            cost_valid, accuracy_valid = sess.run((loss, acc), feed_dict={x:X_valid_c, y:Y_valid})
            print("After {} epochs: {} cost, {} accuracy, {} validation cost, {} validation accuracy".format(epoch, cost, accuracy, cost_valid, accuracy_valid))
            p = np.random.permutation(m)
            X_shuffle = X_train_c[p]
            Y_shuffle = Y_train[p]
            for i in range(0, m, batch_size):
                X_batch = X_shuffle[i:i+batch_size]
                Y_batch = Y_shuffle[i:i+batch_size]
                sess.run(train_op, feed_dict={x:X_batch, y:Y_batch})
        cost, accuracy = sess.run((loss, acc), feed_dict={x:X_train_c, y:Y_train})
        cost_valid, accuracy_valid = sess.run((loss, acc), feed_dict={x:X_valid_c, y:Y_valid})
        print("After {} epochs: {} cost, {} accuracy, {} validation cost, {} validation accuracy".format(epochs, cost, accuracy, cost_valid, accuracy_valid))
        Y_pred = sess.run(y_pred, feed_dict={x:X_valid_c, y:Y_valid})
        print(Y_pred[0])
        Y_pred = np.argmax(Y_pred, 1)
        plt.imshow(X_valid[0])
        plt.title(str(Y_valid[0]) + ' : ' + str(Y_pred[0]))
        plt.show()
```
the output should be
```
2020-06-10 17:45:40.454566: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
After 0 epochs: 3.6953983306884766 cost, 0.09554000198841095 accuracy, 3.6907131671905518 validation cost, 0.09960000216960907 validation accuracy
After 1 epochs: 0.07221613079309464 cost, 0.9774600267410278 accuracy, 0.07810505479574203 validation cost, 0.9764999747276306 validation accuracy
After 2 epochs: 0.0469837561249733 cost, 0.9849399924278259 accuracy, 0.05782172083854675 validation cost, 0.9819999933242798 validation accuracy
After 3 epochs: 0.03443331643939018 cost, 0.9886599779129028 accuracy, 0.050201717764139175 validation cost, 0.9850000143051147 validation accuracy
After 4 epochs: 0.02145254798233509 cost, 0.9930400252342224 accuracy, 0.042915210127830505 validation cost, 0.987500011920929 validation accuracy
After 5 epochs: 0.018733279779553413 cost, 0.9937599897384644 accuracy, 0.05201854929327965 validation cost, 0.9866999983787537 validation accuracy
After 6 epochs: 0.013162973336875439 cost, 0.9957000017166138 accuracy, 0.04360421001911163 validation cost, 0.9883999824523926 validation accuracy
After 7 epochs: 0.023098230361938477 cost, 0.9934200048446655 accuracy, 0.06627936661243439 validation cost, 0.9840999841690063 validation accuracy
After 8 epochs: 0.015782833099365234 cost, 0.9946600198745728 accuracy, 0.054716743528842926 validation cost, 0.9858999848365784 validation accuracy
After 9 epochs: 0.011577955447137356 cost, 0.99617999792099 accuracy, 0.056558288633823395 validation cost, 0.9872000217437744 validation accuracy
After 10 epochs: 0.018797092139720917 cost, 0.9940199851989746 accuracy, 0.06038082763552666 validation cost, 0.9860000014305115 validation accuracy
[9.2108080e-25 7.5373050e-12 2.4517448e-17 1.0000000e+00 4.0947415e-20
 4.7973190e-09 4.8660845e-17 1.2335210e-16 4.6464161e-12 8.3052445e-13]
```
<br>

![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/pictures/lenet5-tensorflow.png)<br>
## Task5 - LeNet-5 (Keras)
Write a function `def lenet5(X):` that builds a modified version of the LeNet-5 architecture using keras:
<br>
* X is a K.Input of shape (m, 28, 28, 1) containing the input images for the network
* * m is the number of images
* The model should consist of the following layers in order:
* Convolutional layer with 6 kernels of shape 5x5 with same padding
* Max pooling layer with kernels of shape 2x2 with 2x2 strides
* Convolutional layer with 16 kernels of shape 5x5 with valid padding
* Max pooling layer with kernels of shape 2x2 with 2x2 strides
* Fully connected layer with 120 nodes
* Fully connected layer with 84 nodes
* Fully connected softmax output layer with 10 nodes
* All layers requiring initialization should initialize their kernels with the he_normal initialization method
* All hidden layers requiring activation should use the relu activation function<br>
you may import tensorflow.keras as K<br>
Returns: a K.Model compiled to use Adam optimization (with default hyperparameters) and accuracy metrics<br>
```
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
lenet5 = __import__('5-lenet5').lenet5

if __name__ == "__main__":
    np.random.seed(0)
    tf.set_random_seed(0)
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    m, h, w = X_train.shape
    X_train_c = X_train.reshape((-1, h, w, 1))
    Y_train = lib['Y_train']
    Y_train_oh = K.utils.to_categorical(Y_train, num_classes=10)
    X_valid = lib['X_valid']
    X_valid_c = X_valid.reshape((-1, h, w, 1))
    Y_valid = lib['Y_valid']
    Y_valid_oh = K.utils.to_categorical(Y_valid, num_classes=10)

    X = K.Input(shape=(h, w, 1))
    model = lenet5(X)
    batch_size = 32
    epochs = 5
    model.fit(X_train_c, Y_train_oh, batch_size=batch_size, epochs=epochs,
                       validation_data=(X_valid_c, Y_valid_oh))

    Y_pred = model.predict(X_valid_c)
    print(Y_pred[0])
    Y_pred = np.argmax(Y_pred, 1)
    plt.imshow(X_valid[0])
    plt.title(str(Y_valid[0]) + ' : ' + str(Y_pred[0]))
    plt.show()
```
the output should be like
```
Train on 50000 samples, validate on 10000 samples
Epoch 1/5
2020-06-10 18:02:43.349202: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
50000/50000 [==============================] - 36s 729us/step - loss: 0.1773 - acc: 0.9456 - val_loss: 0.0729 - val_acc: 0.9797
Epoch 2/5
50000/50000 [==============================] - 35s 709us/step - loss: 0.0642 - acc: 0.9797 - val_loss: 0.0650 - val_acc: 0.9820
Epoch 3/5
50000/50000 [==============================] - 36s 721us/step - loss: 0.0469 - acc: 0.9852 - val_loss: 0.0574 - val_acc: 0.9826
Epoch 4/5
50000/50000 [==============================] - 36s 715us/step - loss: 0.0373 - acc: 0.9882 - val_loss: 0.0472 - val_acc: 0.9847
Epoch 5/5
50000/50000 [==============================] - 36s 718us/step - loss: 0.0300 - acc: 0.9907 - val_loss: 0.0482 - val_acc: 0.9863
[9.2633790e-18 1.3429525e-11 3.5139814e-12 1.0000000e+00 2.3265769e-16
 3.6633261e-09 3.8692485e-14 6.4651492e-12 1.6540311e-12 1.5864870e-13]
```

<br>

![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/pictures/lenet5-tensorflow.png)<br>
