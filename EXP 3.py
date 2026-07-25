import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import math

# ---------------------------------------
# Dynamic Pricing Bandit Environment
# ---------------------------------------
class PricingBanditEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # Five pricing strategies (arms)
        self.prices = [100, 120, 140, 160, 180]

        # Expected revenue probability
        self.success_prob = [0.30, 0.45, 0.60, 0.40, 0.25]

        self.action_space = spaces.Discrete(len(self.prices))
        self.observation_space = spaces.Discrete(1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):

        if np.random.rand() < self.success_prob[action]:
            reward = self.prices[action]
        else:
            reward = 0

        terminated = False
        truncated = False

        return 0, reward, terminated, truncated, {}

# ---------------------------------------
# Epsilon-Greedy
# ---------------------------------------
def epsilon_greedy(env, steps=500, epsilon=0.1):

    Q = np.zeros(env.action_space.n)
    N = np.zeros(env.action_space.n)
    rewards = []

    env.reset()

    for _ in range(steps):

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q)

        _, reward, _, _, _ = env.step(action)

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]

        rewards.append(reward)

    return np.cumsum(rewards)

# ---------------------------------------
# UCB
# ---------------------------------------
def ucb(env, steps=500):

    Q = np.zeros(env.action_space.n)
    N = np.zeros(env.action_space.n)
    rewards = []

    env.reset()

    for t in range(steps):

        if t < env.action_space.n:
            action = t
        else:
            ucb_values = Q + np.sqrt(2 * np.log(t + 1) / N)
            action = np.argmax(ucb_values)

        _, reward, _, _, _ = env.step(action)

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]

        rewards.append(reward)

    return np.cumsum(rewards)

# ---------------------------------------
# Thompson Sampling
# ---------------------------------------
def thompson_sampling(env, steps=500):

    alpha = np.ones(env.action_space.n)
    beta = np.ones(env.action_space.n)

    rewards = []

    env.reset()

    for _ in range(steps):

        samples = np.random.beta(alpha, beta)
        action = np.argmax(samples)

        _, reward, _, _, _ = env.step(action)

        if reward > 0:
            alpha[action] += 1
        else:
            beta[action] += 1

        rewards.append(reward)

    return np.cumsum(rewards)

# ---------------------------------------
# Main Program
# ---------------------------------------
env = PricingBanditEnv()

eps = epsilon_greedy(env)
ucb_rewards = ucb(env)
thompson = thompson_sampling(env)

plt.figure(figsize=(8,5))
plt.plot(eps, label="Epsilon-Greedy")
plt.plot(ucb_rewards, label="UCB")
plt.plot(thompson, label="Thompson Sampling")

plt.xlabel("Pricing Decisions")
plt.ylabel("Cumulative Revenue")
plt.title("Dynamic Pricing using Multi-Armed Bandit")
plt.legend()
plt.grid(True)

plt.show()

print("\nFinal Revenue")
print("Epsilon-Greedy :", eps[-1])
print("UCB            :", ucb_rewards[-1])
print("Thompson       :", thompson[-1])

best = max(
    ("Epsilon-Greedy", eps[-1]),
    ("UCB", ucb_rewards[-1]),
    ("Thompson Sampling", thompson[-1]),
    key=lambda x: x[1]
)

print("\nBest Strategy:", best[0])
