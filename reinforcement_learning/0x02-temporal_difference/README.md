# Monte Carlo, Temporal Difference and Sarsa with reinforcement-learning

## Task0 - Monte Carlo
Write the function def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99): that performs the Monte Carlo algorithm:<br>
<br>
* env is the openAI environment instance
* V is a numpy.ndarray of shape (s,) containing the value estimate
* policy is a function that takes in a state and returns the next action to take
* episodes is the total number of episodes to train over
* max_steps is the maximum number of steps per episode
* alpha is the learning rate
* gamma is the discount rate

Returns: V, the updated value estimate

```
$ cat 0-main.py
#!/usr/bin/env python3

import gym
import numpy as np
monte_carlo = __import__('0-monte_carlo').monte_carlo

np.random.seed(0)

env = gym.make('FrozenLake8x8-v0')
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

def policy(s):
    p = np.random.uniform()
    if p > 0.5:
        if s % 8 != 7 and env.desc[s // 8, s % 8 + 1] != b'H':
            return RIGHT
        elif s // 8 != 7 and env.desc[s // 8 + 1, s % 8] != b'H':
            return DOWN
        elif s // 8 != 0 and env.desc[s // 8 - 1, s % 8] != b'H':
            return UP
        else:
            return LEFT
    else:
        if s // 8 != 7 and env.desc[s // 8 + 1, s % 8] != b'H':
            return DOWN
        elif s % 8 != 7 and env.desc[s // 8, s % 8 + 1] != b'H':
            return RIGHT
        elif s % 8 != 0 and env.desc[s // 8, s % 8 - 1] != b'H':
            return LEFT
        else:
            return UP

V = np.where(env.desc == b'H', -1, 1).reshape(64).astype('float64') 
np.set_printoptions(precision=2)
env.seed(0)
print(monte_carlo(env, V, policy).reshape((8, 8)))

$ ./0-main.py
[[ 0.85  0.85  0.88  0.87  0.86  0.86  0.82  0.83]
 [ 0.86  0.88  0.92  0.94  0.91  0.89  0.86  0.85]
 [ 0.9   0.9   0.95 -1.    0.96  0.92  0.9   0.88]
 [ 0.9   0.91  0.95  0.98  0.97 -1.    0.93  0.93]
 [ 0.93  0.96  0.97 -1.    0.96  0.97  0.95  0.96]
 [ 0.96 -1.   -1.    1.    0.98  0.98 -1.    0.99]
 [ 0.96 -1.    0.98  1.   -1.    0.99 -1.    1.  ]
 [ 0.95  0.97  0.97 -1.    1.    0.99  1.    1.  ]]
```

## Task1 - Temporal Difference with eligibility trace
Write the function def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99): that performs the TD(λ) algorithm:<br>
<br>
* env is the openAI environment instance
* V is a numpy.ndarray of shape (s,) containing the value estimate
* policy is a function that takes in a state and returns the next action to take
* lambtha is the eligibility trace factor
* episodes is the total number of episodes to train over
* max_steps is the maximum number of steps per episode
* alpha is the learning rate
* gamma is the discount rate

Returns: V, the updated value estimate

```
$ cat 1-main.py
#!/usr/bin/env python3

import gym
import numpy as np
td_lambtha = __import__('1-td_lambtha').td_lambtha

np.random.seed(0)

env = gym.make('FrozenLake8x8-v0')
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

def policy(s):
    p = np.random.uniform()
    if p > 0.5:
        if s % 8 != 7 and env.desc[s // 8, s % 8 + 1] != b'H':
            return RIGHT
        elif s // 8 != 7 and env.desc[s // 8 + 1, s % 8] != b'H':
            return DOWN
        elif s // 8 != 0 and env.desc[s // 8 - 1, s % 8] != b'H':
            return UP
        else:
            return LEFT
    else:
        if s // 8 != 7 and env.desc[s // 8 + 1, s % 8] != b'H':
            return DOWN
        elif s % 8 != 7 and env.desc[s // 8, s % 8 + 1] != b'H':
            return RIGHT
        elif s % 8 != 0 and env.desc[s // 8, s % 8 - 1] != b'H':
            return LEFT
        else:
            return UP

V = np.where(env.desc == b'H', -1, 1).reshape(64).astype('float64') 
np.set_printoptions(precision=4)
print(td_lambtha(env, V, policy, 0.9).reshape((8, 8)))

$ ./1-main.py
[[-0.8329 -0.8576 -0.8437 -0.8352 -0.8458 -0.7143 -0.6903 -0.6944]
 [-0.8761 -0.8864 -0.8701 -0.8821 -0.8314 -0.7741 -0.7312 -0.7335]
 [-0.8837 -0.8917 -0.9139 -1.     -0.8525 -0.7828 -0.6618 -0.5681]
 [-0.9014 -0.9162 -0.9396 -0.979  -0.9685 -1.     -0.7332 -0.5125]
 [-0.9126 -0.9239 -0.925  -1.     -0.9445 -0.9099 -0.8717 -0.579 ]
 [-0.9668 -1.     -1.      0.801  -0.9277 -0.8891 -1.     -0.283 ]
 [-0.9412 -1.     -0.6664  0.4184 -1.     -0.4697 -1.      0.6113]
 [-0.9164 -0.921  -0.9134 -1.      1.      0.2385  0.4616  1.    ]]
```

## Task2 - Sarsa with eligibility trace
Write the function def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05): that performs SARSA(λ):<br>
<br>
* env is the openAI environment instance
* Q is a numpy.ndarray of shape (s,a) containing the Q table
* lambtha is the eligibility trace factor
* episodes is the total number of episodes to train over
* max_steps is the maximum number of steps per episode
* alpha is the learning rate
* gamma is the discount rate
* epsilon is the initial threshold for epsilon greedy
* min_epsilon is the minimum value that epsilon should decay to
* epsilon_decay is the decay rate for updating epsilon between episodes

Returns: Q, the updated Q table

```
$ cat 2-main.py
#!/usr/bin/env python3

import gym
import numpy as np
sarsa_lambtha = __import__('2-sarsa_lambtha').sarsa_lambtha

np.random.seed(0)
env = gym.make('FrozenLake8x8-v0')
Q = np.random.uniform(size=(64, 4))
np.set_printoptions(precision=4)
print(sarsa_lambtha(env, Q, 0.9))

$ ./2-main.py
[[0.5438 0.7152 0.6028 0.5449]
 [0.4237 0.6459 0.4376 0.8918]
 [0.9637 0.3834 0.7917 0.5289]
 [0.568  0.9256 0.071  0.0871]
 [0.0202 0.8326 0.7782 0.87  ]
 [0.9786 0.7992 0.4615 0.7805]
 [0.1183 0.6399 0.1434 0.9447]
 [0.5218 0.4147 0.2646 0.7742]
 [0.4562 0.5684 0.0188 0.605 ]
 [0.6121 0.6169 0.894  0.6818]
 [0.3595 0.4385 0.6976 0.0602]
 [0.6668 0.6706 0.2104 0.1289]
 [0.3154 0.3637 0.5702 0.4386]
 [0.9884 0.102  0.2089 0.1613]
 [0.6531 0.2533 0.4663 0.2444]
 [0.159  0.1104 0.6563 0.1382]
 [0.1966 0.3687 0.821  0.0971]
 [0.8379 0.0961 0.9765 0.4687]
 [0.9768 0.6048 0.7079 0.0392]
 [0.2828 0.1202 0.2961 0.1187]
 [0.318  0.4143 0.0641 0.6925]
 [0.5666 0.2654 0.5232 0.0939]
 [0.5759 0.9293 0.3186 0.6674]
 [0.1318 0.7163 0.2894 0.1832]
 [0.5865 0.0201 0.8289 0.0047]
 [0.6778 0.27   0.7352 0.9622]
 [0.2488 0.5599 0.592  0.5723]
 [0.2231 0.9527 0.4471 0.8464]
 [0.6995 0.2974 0.8138 0.3965]
 [0.8811 0.5813 0.8817 0.6925]
 [0.7253 0.5013 0.9561 0.644 ]
 [0.4239 0.6064 0.0192 0.3016]
 [0.6602 0.2901 0.618  0.4288]
 [0.1355 0.2983 0.57   0.5909]
 [0.5569 0.6532 0.6521 0.4314]
 [0.8965 0.3676 0.4359 0.8919]
 [0.8062 0.7039 0.1002 0.9195]
 [0.7142 0.9988 0.1494 0.8681]
 [0.1625 0.6156 0.1238 0.848 ]
 [0.8073 0.5691 0.4072 0.0692]
 [0.6974 0.4535 0.7221 0.8664]
 [0.9755 0.8558 0.0117 0.36  ]
 [0.4053 0.3335 0.3524 0.3312]
 [0.2    0.0185 0.7937 0.2239]
 [0.3454 0.9281 0.7044 0.0318]
 [0.1647 0.6215 0.5772 0.2379]
 [0.9342 0.614  0.5356 0.5899]
 [0.7301 0.3119 0.3982 0.2098]
 [0.1862 0.9444 0.7396 0.4905]
 [0.2274 0.2544 0.058  0.4344]
 [0.3118 0.6963 0.3778 0.1796]
 [0.0247 0.0672 0.6794 0.4537]
 [0.5366 0.8967 0.9903 0.2169]
 [0.6631 0.2633 0.0207 0.7584]
 [0.32   0.3835 0.5883 0.831 ]
 [0.629  0.8727 0.2735 0.798 ]
 [0.1856 0.9528 0.6875 0.2155]
 [0.9474 0.7309 0.2539 0.2133]
 [0.5182 0.0257 0.2075 0.4247]
 [0.3742 0.4636 0.2776 0.5868]
 [0.8639 0.1175 0.5174 0.1321]
 [0.7169 0.3961 0.5654 0.1833]
 [0.1448 0.4881 0.3556 0.9404]
 [0.7653 0.7487 0.9037 0.0834]]
```
