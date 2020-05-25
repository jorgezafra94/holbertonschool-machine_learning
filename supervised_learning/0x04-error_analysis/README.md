# Error Analisys
## Task0 - confusion Matrix
Write the function `def create_confusion_matrix(labels, logits):` that creates a confusion matrix
```
  #!/usr/bin/env python3

  import numpy as np
  create_confusion_matrix = __import__('0-create_confusion').create_confusion_matrix  

  if __name__ == '__main__':
      lib = np.load('labels_logits.npz')
      labels = lib['labels']
      logits = lib['logits']

      np.set_printoptions(suppress=True)
      confusion = create_confusion_matrix(labels, logits)
      print(confusion)
      np.savez_compressed('confusion.npz', confusion=confusion)
 ```
 you should get
 ```
  ubuntu-xenial:0x04-error_analysis$ ./0-main.py 
  [[4701.    0.   36.   17.   12.   81.   38.   11.   35.    1.]
  [   0. 5494.   36.   21.    3.   38.    7.   13.   59.    7.]
  [  64.   93. 4188.  103.  108.   17.  162.   80.  132.   21.]
  [  30.   48.  171. 4310.    2.  252.   22.   86.  128.   52.]
  [  17.   27.   35.    0. 4338.   11.   84.    9.   27.  311.]
  [  89.   57.   45.  235.   70. 3631.  123.   33.  163.   60.]
  [  47.   32.   87.    1.   64.   83. 4607.    0.   29.    1.]
  [  26.   95.   75.    7.   58.   18.    1. 4682.   13.  200.]
  [  31.  153.   82.  174.   27.  179.   64.    7. 4003.  122.]
  [  48.   37.   39.   71.  220.   49.    8.  244.   46. 4226.]]
  ubuntu-xenial:0x04-error_analysis$
```

## Task1 - Sensitivity
Write the function `def sensitivity(confusion):` that calculates the sensitivity for each class in a confusion matrix
```
 #!/usr/bin/env python3

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(sensitivity(confusion))
 ```
 you should get
 ```
 ubuntu-xenial:0x04-error_analysis$ ./1-main.py 
[0.95316302 0.96759422 0.84299517 0.84493237 0.89277629 0.80581447
 0.93051909 0.9047343  0.82672449 0.84723336]
```

## Task2 - Precision
Write the function `def precision(confusion):` that calculates the precision for each class in a confusion matrix
```
#!/usr/bin/env python3

import numpy as np
precision = __import__('2-precision').precision

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(precision(confusion))
 ```
 you should get
 ```
 ubuntu-xenial:0x04-error_analysis$ ./2-main.py 
[0.93033841 0.91020543 0.87359199 0.87264628 0.88494492 0.83298922
 0.90050821 0.90648596 0.86364617 0.84503099]
```

## Task3 - Specificity
Write the function `def specificity(confusion):` that calculates the specificity for each class in a confusion matrix
```
 #!/usr/bin/env python3

import numpy as np
specificity = __import__('3-specificity').specificity

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(specificity(confusion))
 ```
 you should get
 ```
 ubuntu-xenial:0x04-error_analysis$ ./3-main.py 
[0.99218958 0.98777131 0.9865429  0.98599078 0.98750582 0.98399789
 0.98870119 0.98922476 0.98600469 0.98278237]
```

## Task4 - f1-score
Write the function `def f1_score(confusion):` that calculates the F1 score of a confusion matrix
```
 #!/usr/bin/env python3

import numpy as np
f1_score = __import__('4-f1_score').f1_score

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(f1_score(confusion))
 ```
 you should get
 ```
ubuntu-xenial:0x04-error_analysis$ ./4-main.py 
[0.94161242 0.93802288 0.8580209  0.85856574 0.88884336 0.81917654
 0.91526771 0.90560928 0.8447821  0.84613074]
```
## Task5 Answer
In the text file 5-error_handling, write the lettered answer to the question of how you should approach the following scenarios. Please write the answer to each scenario on a different line. If there is more than one way to approach a scenario, please use CSV formatting and place your answers in alphabetical order (ex. A,B,C):

Scenarios:

1. High Bias, High Variance
2. High Bias, Low Variance
3. Low Bias, High Variance
4. Low Bias, Low Variance
Approaches:

A. Train more
B. Try a different architecture
C. Get more data
D. Build a deeper network
E. Use regularization
F. Nothing

