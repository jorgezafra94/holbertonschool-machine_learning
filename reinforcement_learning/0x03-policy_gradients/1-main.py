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
# state = [[0.04228739, -0.04522399,  0.01190918, -0.03496226]]
# state = np.array(state)
print(weight)
print(state)

action, grad = policy_gradient(state, weight)
print(action)
print(grad)

env.close()