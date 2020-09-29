#!/usr/bin/env python3
"""
based on:
https://www.kaggle.com/charel/learn-by-example-reinforcement-learning-with-gym

play with the model trained with Q-learning
"""

import numpy as np


def play(env, Q, max_steps=100):
    """
    * env is the FrozenLakeEnv instance
    * Q is a numpy.ndarray containing the Q-table
    * max_steps is the maximum number of steps in the episode
    * Each state of the board should be displayed via the console
    * You should always exploit the Q-table

    Returns: the total rewards for the episode
    """
    state = env.reset()
    env.render()
    done = False
    for step in range(max_steps):
        action = np.argmax(Q[state])
        new_state, reward, done, info = env.step(action)
        if done is True:
            env.render()
            print(reward)
            break
        else:
            env.render()
            state = new_state
