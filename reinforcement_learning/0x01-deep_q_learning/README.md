# Breakout
Write a python script train.py that utilizes keras, keras-rl, and gym to train an agent that can play Atari’s Breakout:
<br>
* Your script should utilize keras-rl‘s DQNAgent, SequentialMemory, and EpsGreedyQPolicy
* Your script should save the final policy network as policy.h5

Write a python script play.py that can display a game played by the agent trained by train.py:
<br>
* Your script should load the policy network saved in policy.h5
* Your agent should use the GreedyQPolicy

To run the train.py file
```
./train.py
```
this file is going to create a policy.h5 file that is going to be loaded in the play.py file

To play with the game
```
./play.py
```
