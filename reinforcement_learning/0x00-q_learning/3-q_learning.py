#!/usr/bin/env python3
"""
based on:
https://www.youtube.com/watch?v=HGeI30uATws&feature=youtu.
be&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&t=24

and

https://www.kaggle.com/charel/learn-by-example-reinforcement-learning-with-gym

here we got the code to realize the Q-learning
the only part that I change was in the done is True part
"""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
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
    """

    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        # reward current episode
        reward_c_e = 0

        for step in range(max_steps):
            # Exploration-explotation trade-off

            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            # Update Q-table for Q(s, a)
            part = (reward + gamma * np.max(Q[new_state]) - Q[state, action])
            Q[state, action] += alpha * part
            state = new_state

            if done is True:
                if reward == 0.0:
                    # if falls in a hole or lost last live
                    reward_c_e = -1
                reward_c_e += reward
                break

            reward_c_e += reward
        total_rewards.append(reward_c_e)
        part = (1 - min_epsilon) * np.exp(-epsilon_decay * episode)
        epsilon = min_epsilon + part
    return Q, total_rewards
