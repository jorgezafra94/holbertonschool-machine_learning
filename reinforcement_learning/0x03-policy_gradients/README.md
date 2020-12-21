# Policy Gradient

## Task0 - Simple Policy function
Write a function that computes to policy with a weight of a matrix.<br>
<br>
* Prototype: def policy(matrix, weight):

```
$ cat 0-main.py
#!/usr/bin/env python3
"""
Main file
"""
import numpy as np
from policy_gradient import policy


weight = np.ndarray((4, 2), buffer=np.array([
    [4.17022005e-01, 7.20324493e-01], 
    [1.14374817e-04, 3.02332573e-01], 
    [1.46755891e-01, 9.23385948e-02], 
    [1.86260211e-01, 3.45560727e-01]
    ]))
state = np.ndarray((1, 4), buffer=np.array([
    [-0.04428214,  0.01636746,  0.01196594, -0.03095031]
    ]))

res = policy(state, weight)
print(res)

$
$ ./0-main.py
[[0.50351642 0.49648358]]
$
```

## Task1 - Compute the Monte-Carlo policy gradient
By using the previous function created policy, write a function that computes the Monte-Carlo policy gradient based on a state and a weight matrix.<br>
<br>
* Prototype: def policy_gradient(state, weight):
* state: matrix representing the current observation of the environment
* weight: matrix of random weight

Return: the action and the gradient (in this order)

```
$ cat 1-main.py
#!/usr/bin/env python3
"""
Main file
"""
import gym
import numpy as np
from policy_gradient import policy_gradient

env = gym.make('CartPole-v1')
np.random.seed(1)

weight = np.random.rand(4, 2)
state = env.reset()[None,:]
print(weight)
print(state)

action, grad = policy_gradient(state, weight)
print(action)
print(grad)

env.close()

$ 
$ ./1-main.py
[[4.17022005e-01 7.20324493e-01]
 [1.14374817e-04 3.02332573e-01]
 [1.46755891e-01 9.23385948e-02]
 [1.86260211e-01 3.45560727e-01]]
[[ 0.04228739 -0.04522399  0.01190918 -0.03496226]]
0
[[ 0.02106907 -0.02106907]
 [-0.02253219  0.02253219]
 [ 0.00593357 -0.00593357]
 [-0.01741943  0.01741943]]
$ 
```

## Task2 - Implement the training
By using the previous function created policy_gradient, write a function that implements a full training.<br>
<br>
* Prototype: def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
* env: initial environment
* nb_episodes: number of episodes used for training
* alpha: the learning rate
* gamma: the discount factor

Return: all values of the score (sum of all rewards during one episode loop)<br>
Since the training is quite long, please print the current episode number and the score after each loop. To display these information on the same line, you can use end="\r", flush=False of the print function.<br>
<br>
With the following main file, you should have this result plotted:<br>
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/reinforcement_learning/0x03-policy_gradients/image1.PNG)
<br>

```
$ cat 2-main.py
#!/usr/bin/env python3
"""
Main file
"""
import gym
import matplotlib.pyplot as plt
import numpy as np

from train import train

env = gym.make('CartPole-v1')

scores = train(env, 10000)

plt.plot(np.arange(len(scores)), scores)
plt.show()
env.close()

$ 
$ ./2-main.py
```

## Task3 - Animate iteration
Update the prototype of the train function by adding a last optional parameter show_result (default: False).<br>
<br>
* When this parameter is True, render the environment every 1000 episodes computed.

```
$ cat 3-main.py
#!/usr/bin/env python3
"""
Main file
"""
import gym

from train import train

env = gym.make('CartPole-v1')

scores = train(env, 10000, 0.000045, 0.98, True)

env.close()

$ 
$ ./3-main.py
```
