#!/usr/bin/env python3
"""
* https://towardsdatascience.com/eligibility-traces
  in-reinforcement-learning-a6b458c019d6
Sarsa with eligibility trace
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    * Q is a numpy.ndarray containing the q-table
    * state is the current state
    * epsilon is the epsilon to use for the calculation
    * You should sample p with numpy.random.uniformn to
      determine if your algorithm should explore or exploit
    * If exploring, you should pick the next action with
      numpy.random.randint from all possible actions

    Returns: the next action index
    """
    if np.random.uniform(0, 1) < epsilon:
        # Explore action space randomly
        # the shape of Q-table is (num of states, num of actions)
        action = np.random.randint(0, Q.shape[1])
    else:
        # Exploit learned values by choosing optimal values
        # as we know states are rows columns actions
        action = np.argmax(Q[state])

    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    * env: the openAI environment instance
    * Q: np.ndarray of shape(s,a) containing the Qtable
    * lambtha: the eligibility trace factor
    * episodes: the total number of episodes to train over
    * max_steps: the maximum number of steps per episode
    * alpha: the learning rate
    * gamma: the discount rate
    * epsilon: initial threshold for epsilon greedy
    * min_epsilon: the minimum value that epsilon should decay to
    * epsilon_decay the devay rate for updating epsilon between episodes
    Returns: Q, the updated Q table
    """
    for i in range(episodes):
        state = env.reset()
        e = np.zeros((Q.shape))
        action = epsilon_greedy(Q, state, epsilon=epsilon)
        for j in range(max_steps):
            new_state, reward, done, _ = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon=epsilon)
            f = reward + gamma * Q[new_state, new_action] - Q[state, action]
            e[state, action] += 1

            Q[state, action] = Q[state, action] + alpha * f * e[state, action]
            state = new_state
            action = new_action

        if done:
            break
        part = (1 - min_epsilon) * np.exp(-epsilon_decay * i)
        epsilon = min_epsilon + part

    return Q
