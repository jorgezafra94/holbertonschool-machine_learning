#!/usr/bin/env python3
"""
based on:
https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
Training Atari breakout
"""

import gym
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
import keras as K
import numpy as np

create_q_model = __import__('train').my_model
AtariProcessor = __import__('train').AtariProcessor


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n
    window = 4

    model = create_q_model(nb_actions, window)
    memory = SequentialMemory(limit=1000000, window_length=window)
    processor = AtariProcessor()

    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   processor=processor,
                   memory=memory)

    dqn.compile(K.optimizers.Adam(lr=.00025), metrics=['mae'])

    # load weights.
    dqn.load_weights('policy.h5')

    # evaluate algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=True)
