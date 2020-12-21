#!/usr/bin/env python3
"""
Training reinforcement model using policy gradient
"""

import numpy as np
import matplotlib.pyplot as plt


def policy(matrix, weight):
    """
    * matrix: states
    * weights: weights of the states
    Return: the new policy
    """

    result = np.dot(matrix, weight)
    result = np.exp(result)
    my_policy = result / result.sum()
    return my_policy


def policy_gradient(state, probs, action):
    """
    We have to change the function
    because we need to implement the nA env.action_space.n
    * state: matrix representing the current observation of
      the environment
    * weight: matrix of random weight
    Return: the action and the gradient
    """

    def softmax_grad(softmax):
        s = softmax.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    dsoftmax = softmax_grad(probs)[action, :]
    dlog = dsoftmax / probs[0, action]
    grad = state.T.dot(dlog[None, :])
    return grad


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    * env: initial environment
    * nb_episodes: number of episodes used for training
    * alpha: the learning rate
    * gamma: the discount factor

    Return: all values of the score
    (sum of all rewards during one episode loop)
    """
    weight = np.random.rand(4, 2)
    nA = env.action_space.n
    episode_rewards = []
    for e in range(nb_episodes):
        state = env.reset()[None, :]
        grads = []
        rewards = []
        score = 0

        while True:
            if show_result and not e % 1000:
                env.render()
            probs = policy(state, weight)
            action = np.random.choice(nA, p=probs[0])
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[None, :]

            grad = policy_gradient(state, probs, action)
            grads.append(grad)

            grads.append(grad)
            rewards.append(reward)

            score += reward
            state = next_state

            if done:
                break

        for i in range(len(grads)):
            aux = sum([r * (gamma ** r) for t, r in enumerate(rewards[i:])])
            weight += alpha * grads[i] * aux

        episode_rewards.append(score)
        print("EP: " + str(e) + " Score: " + str(score) + "        ",
              end="\r", flush=False)

    return episode_rewards
