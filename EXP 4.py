import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Delivery Drone Environment
# -----------------------------
class DeliveryDroneEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.grid_size = 5

        # Warehouse (Start)
        self.start = (0, 0)

        # Delivery points (Rewards)
        self.goals = [(4, 4), (2, 3)]

        self.state = self.start

        # Actions
        # 0 = Up, 1 = Down, 2 = Left, 3 = Right
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(self.grid_size * self.grid_size)

    def state_to_index(self, state):
        return state[0] * self.grid_size + state[1]

    def index_to_state(self, index):
        return (index // self.grid_size, index % self.grid_size)

    def reset(self, seed=None, options=None):
        self.state = self.start
        return self.state_to_index(self.state), {}

    def step(self, action):

        r, c = self.state

        if action == 0:      # Up
            r = max(0, r - 1)
        elif action == 1:    # Down
            r = min(self.grid_size - 1, r + 1)
        elif action == 2:    # Left
            c = max(0, c - 1)
        elif action == 3:    # Right
            c = min(self.grid_size - 1, c + 1)

        self.state = (r, c)

        reward = -1
        done = False

        if self.state in self.goals:
            reward = 20
            done = True

        return self.state_to_index(self.state), reward, done, False, {}

# ------------------------------------
# Initialize Environment
# ------------------------------------
env = DeliveryDroneEnv()

num_states = env.observation_space.n
num_actions = env.action_space.n

gamma = 0.9

# Random Policy
policy = np.random.randint(num_actions, size=num_states)

# State Values
V = np.zeros(num_states)

# ------------------------------------
# Policy Iteration
# ------------------------------------
stable = False

while not stable:

    # Policy Evaluation
    while True:

        delta = 0

        for s in range(num_states):

            env.state = env.index_to_state(s)

            action = policy[s]

            next_state, reward, done, _, _ = env.step(action)

            value = reward

            if not done:
                value += gamma * V[next_state]

            delta = max(delta, abs(V[s] - value))
            V[s] = value

        if delta < 1e-4:
            break

    # Policy Improvement
    stable = True

    for s in range(num_states):

        old_action = policy[s]

        action_values = []

        for a in range(num_actions):

            env.state = env.index_to_state(s)

            next_state, reward, done, _, _ = env.step(a)

            value = reward

            if not done:
                value += gamma * V[next_state]

            action_values.append(value)

        policy[s] = np.argmax(action_values)

        if old_action != policy[s]:
            stable = False

# ------------------------------------
# Print Optimal Policy
# ------------------------------------
symbols = ['↑', '↓', '←', '→']

print("\nOptimal Policy:\n")

for r in range(env.grid_size):

    for c in range(env.grid_size):

        if (r, c) in env.goals:
            print(" G ", end=" ")
        else:
            s = env.state_to_index((r, c))
            print(symbols[policy[s]], end=" ")

    print()

# ------------------------------------
# Display Value Function
# ------------------------------------
values = V.reshape((env.grid_size, env.grid_size))

plt.imshow(values, cmap="viridis")
plt.colorbar(label="State Value")
plt.title("Optimal State Values")
plt.show()
