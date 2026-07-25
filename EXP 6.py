import numpy as np
import random

# True click probabilities of 3 ads
true_ctr = [0.2, 0.5, 0.8]

n_ads = len(true_ctr)
rounds = 1000

# Function to simulate click
def get_reward(ad):
    return 1 if random.random() < true_ctr[ad] else 0

# ---------------- Epsilon-Greedy ----------------
epsilon = 0.1
counts = [0] * n_ads
values = [0] * n_ads
reward_eg = 0

for t in range(rounds):
    if random.random() < epsilon:
        ad = random.randint(0, n_ads - 1)
    else:
        ad = np.argmax(values)

    reward = get_reward(ad)
    reward_eg += reward

    counts[ad] += 1
    values[ad] += (reward - values[ad]) / counts[ad]

# ---------------- UCB ----------------
counts = [0] * n_ads
values = [0] * n_ads
reward_ucb = 0

for t in range(rounds):
    if t < n_ads:
        ad = t
    else:
        ucb = [values[i] + np.sqrt(2 * np.log(t + 1) / counts[i]) for i in range(n_ads)]
        ad = np.argmax(ucb)

    reward = get_reward(ad)
    reward_ucb += reward

    counts[ad] += 1
    values[ad] += (reward - values[ad]) / counts[ad]

# ---------------- Thompson Sampling ----------------
success = [1] * n_ads
failure = [1] * n_ads
reward_ts = 0

for t in range(rounds):
    samples = [np.random.beta(success[i], failure[i]) for i in range(n_ads)]
    ad = np.argmax(samples)

    reward = get_reward(ad)
    reward_ts += reward

    if reward == 1:
        success[ad] += 1
    else:
        failure[ad] += 1

# ---------------- Results ----------------
print("Epsilon-Greedy CTR:", reward_eg / rounds)
print("UCB CTR:", reward_ucb / rounds)
print("Thompson Sampling CTR:", reward_ts / rounds)
