#!/usr/bin/env python3
"""
* https://amreis.github.io/ml/reinf-learn/2017/11/02
  reinforcement-learning-eligibility-traces.html
* https://towardsdatascience.com/eligibility-traces-
  in-reinforcement-learning-a6b458c019d6
Temporal Difference
"""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):

    eligibility = np.zeros(env.observation_space.n)
    for i in range(episodes):
        state = env.reset()
        for j in range(max_steps):
            action = policy(state)
            new_state, reward, done, _ = env.step(action)

            td_error = reward + gamma * V[new_state] - V[state]
            eligibility[state] += 1.0

            V = V + alpha * td_error * eligibility
            eligibility *= lambtha * gamma

            if done:
                break
            else:
                state = new_state

    return V
