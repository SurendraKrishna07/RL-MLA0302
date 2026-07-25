import random

episodes = 1000

# Success probability of representatives
success_rate = [0.6, 0.8]   # Rep1, Rep2

# Monte Carlo Simulation
def monte_carlo(policy):
    total_reward = 0

    for _ in range(episodes):
        rep = policy()

        if random.random() < success_rate[rep]:
            reward = 1      # Call resolved
        else:
            reward = 0      # Call not resolved

        total_reward += reward

    return total_reward / episodes

# Policy 1: Random Assignment
def random_policy():
    return random.randint(0, 1)

# Policy 2: Always assign to experienced representative
def experienced_policy():
    return 1

# Estimate value functions
value_random = monte_carlo(random_policy)
value_experienced = monte_carlo(experienced_policy)

print("Random Policy Value :", round(value_random, 2))
print("Experienced Policy Value :", round(value_experienced, 2))

if value_experienced > value_random:
    print("Experienced Policy performs better.")
else:
    print("Random Policy performs better.")
