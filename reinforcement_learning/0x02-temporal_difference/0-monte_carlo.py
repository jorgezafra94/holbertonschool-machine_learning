#!/usr/bin/env python3
"""
* https://livebook.manning.com/book/grokking-deep-reinforcement-
  learning/chapter-3/v-4/74
* https://medium.com/deep-math-machine-learning-ai/ch-12-1-model
  -free-reinforcement-learning-algorithms-monte-carlo-sarsa-q-
  learning-65267cb8d1b4
Monte Carlo Eveluate Policy
"""

import numpy as np


def generate_episode(env, policy, max_steps):
    """
    the idea with this funciton is to generate each episode
    * env is the openAI environment instance
    * max_steps is the maximum number of steps per episode
    * policy is a function that takes in a state and returns
      the next action to take
    """
    episodes = []
    state = env.reset()
    for t in range(max_steps):
        action = policy(state)
        new_state, reward, done, _ = env.step(action)

        if done is True:
            episodes.append((state, action, 1))
            break
        episodes.append((state, action, reward))
        state = new_state
    return np.array(episodes, np.object)


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    * env is the openAI environment instance
    * V is a numpy.ndarray of shape (s,) containing the value
      estimate
    * policy is a function that takes in a state and returns
      the next action to take
    * episodes is the total number of episodes to train over
    * max_steps is the maximum number of steps per episode
    * alpha is the learning rate
    * gamma is the discount rate
    Returns: V, the updated value estimate
    """
    nS = env.observation_space.n
    discounts = np.logspace(0, max_steps, num=max_steps,
                            base=gamma, endpoint=False)
    for i in range(episodes):
        # alpha = max(alpha * np.exp(-0.01 * i), 0.0)
        episode = generate_episode(env, policy, max_steps)
        return_visited = np.zeros(nS, dtype=bool)
        for step, (state, action, reward) in enumerate(episode):
            if return_visited[state]:
                continue
            return_visited[state] = True
            seq_len = len(episode[step:])
            G = np.sum(discounts[:seq_len] * episode[step:, 2])
            V[state] = V[state] + alpha * (G - V[state])
    return V
