# Q-learning
## Task0 - Load the Environment
Write a function def load_frozen_lake(desc=None, map_name=None, is_slippery=False): that loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym:<br>
<br>
* desc is either None or a list of lists containing a custom description of the map to load for the environment
* map_name is either None or a string containing the pre-made map to load
* Note: If both desc and map_name are None, the environment will load a randomly generated 8x8 map
* is_slippery is a boolean to determine if the ice is slippery

Returns: the environment

```
$ cat 0-main.py
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
import numpy as np

np.random.seed(0)
env = load_frozen_lake()
print(env.desc)
print(env.P[0][0])
env = load_frozen_lake(is_slippery=True)
print(env.desc)
print(env.P[0][0])
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
print(env.desc)
env = load_frozen_lake(map_name='4x4')
print(env.desc)
$ ./0-main.py
[[b'S' b'F' b'F' b'F' b'F' b'F' b'F' b'H']
 [b'H' b'F' b'F' b'F' b'F' b'H' b'F' b'F']
 [b'F' b'H' b'F' b'H' b'H' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'H' b'F' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'F' b'H' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'F' b'H' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'F' b'F' b'G']]
[(1.0, 0, 0.0, False)]
[[b'S' b'F' b'H' b'F' b'H' b'F' b'H' b'F']
 [b'H' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'H' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'F' b'H' b'F' b'F' b'F' b'F' b'H']
 [b'F' b'F' b'F' b'F' b'F' b'H' b'F' b'H']
 [b'F' b'F' b'H' b'F' b'H' b'F' b'H' b'F']
 [b'F' b'F' b'H' b'F' b'F' b'F' b'F' b'G']]
[(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 8, 0.0, True)]
[[b'S' b'F' b'F']
 [b'F' b'H' b'H']
 [b'F' b'F' b'G']]
[[b'S' b'F' b'F' b'F']
 [b'F' b'H' b'F' b'H']
 [b'F' b'F' b'F' b'H']
 [b'H' b'F' b'F' b'G']]
$
```

## Task1 - Initialize Q-table
Write a function def q_init(env): that initializes the Q-table:<br>
<br>
* env is the FrozenLakeEnv instance

Returns: the Q-table as a numpy.ndarray of zeros

```
$ cat 1-main.py
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init

env = load_frozen_lake()
Q = q_init(env)
print(Q.shape)
env = load_frozen_lake(is_slippery=True)
Q = q_init(env)
print(Q.shape)
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)
print(Q.shape)
env = load_frozen_lake(map_name='4x4')
Q = q_init(env)
print(Q.shape)
$ ./1-main.py
(64, 4)
(64, 4)
(9, 4)
(16, 4)
$
```

## Task2 - Epsilon Greedy
Write a function def epsilon_greedy(Q, state, epsilon): that uses epsilon-greedy to determine the next action:<br>
<br>
* Q is a numpy.ndarray containing the q-table
* state is the current state
* epsilon is the epsilon to use for the calculation
* You should sample p with numpy.random.uniformn to determine if your algorithm should explore or exploit
* If exploring, you should pick the next action with numpy.random.randint from all possible actions

Returns: the next action index

```
$ cat 2-main.py
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy
import numpy as np

desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)
Q[7] = np.array([0.5, 0.7, 1, -1])
np.random.seed(0)
print(epsilon_greedy(Q, 7, 0.5))
np.random.seed(1)
print(epsilon_greedy(Q, 7, 0.5))
$ ./2-main.py
2
0
$
```

## task3 - Q-learning
Write the function def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05): that performs Q-learning:<br>
<br>
* env is the FrozenLakeEnv instance
* Q is a numpy.ndarray containing the Q-table
* episodes is the total number of episodes to train over
* max_steps is the maximum number of steps per episode
* alpha is the learning rate
* gamma is the discount rate
* epsilon is the initial threshold for epsilon greedy
* min_epsilon is the minimum value that epsilon should decay to
* epsilon_decay is the decay rate for updating epsilon between episodes
* When the agent falls in a hole, the reward should be updated to be -1

Returns: Q, total_rewards
* Q is the updated Q-table
* total_rewards is a list containing the rewards per episode

```
$ cat 3-main.py
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
train = __import__('3-q_learning').train
import numpy as np

np.random.seed(0)
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)

Q, total_rewards  = train(env, Q)
print(Q)
split_rewards = np.split(np.array(total_rewards), 10)
for i, rewards in enumerate(split_rewards):
    print((i+1) * 500, ':', np.mean(rewards))
$ ./3-main.py
[[0.96059593 0.970299   0.95098488 0.96059396]
 [0.96059557 0.         0.0094072  0.37627228]
 [0.18061285 0.         0.         0.        ]
 [0.97029877 0.9801     0.         0.96059583]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.98009763 0.98009933 0.99       0.9702983 ]
 [0.98009922 0.98999782 1.         0.        ]
 [0.         0.         0.         0.        ]]
500 : 0.812
1000 : 0.88
1500 : 0.9
2000 : 0.9
2500 : 0.88
3000 : 0.844
3500 : 0.892
4000 : 0.896
4500 : 0.852
5000 : 0.928
```

## Task4 - Play
Write a function def play(env, Q, max_steps=100): that has the trained agent play an episode:<br>
<br>
* env is the FrozenLakeEnv instance
* Q is a numpy.ndarray containing the Q-table
* max_steps is the maximum number of steps in the episode
* Each state of the board should be displayed via the console
* You should always exploit the Q-table

Returns: the total rewards for the episode

```
$ cat 4-main.py
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
train = __import__('3-q_learning').train
play = __import__('4-play').play

import numpy as np

np.random.seed(0)
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)

Q, total_rewards  = train(env, Q)
print(play(env, Q))
$ ./4-main.py
'S'FF
FHH
FFG
  (Down)
SFF
'F'HH
FFG
  (Down)
SFF
FHH
'F'FG
  (Right)
SFF
FHH
F'F'G
  (Right)
SFF
FHH
FF'G'
1.0
None

```
