import gymnasium as gym
from gymnasium import spaces
import numpy as np

# -------------------------------
# Warehouse Environment
# -------------------------------
class WarehouseEnv(gym.Env):

    def __init__(self):
        super(WarehouseEnv, self).__init__()

        self.size = 5

        # Actions:
        # 0 = Up
        # 1 = Down
        # 2 = Left
        # 3 = Right
        self.action_space = spaces.Discrete(4)

        # Observation = Agent Position
        self.observation_space = spaces.MultiDiscrete([self.size, self.size])

        self.start = (0, 0)
        self.goal = (4, 4)

        self.items = [(2, 2)]
        self.obstacles = [(1, 1), (3, 2)]

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent = list(self.start)
        self.item_collected = False
        return np.array(self.agent), {}

    def step(self, action):

        row, col = self.agent

        # Move Agent
        if action == 0 and row > 0:
            row -= 1
        elif action == 1 and row < self.size - 1:
            row += 1
        elif action == 2 and col > 0:
            col -= 1
        elif action == 3 and col < self.size - 1:
            col += 1

        reward = 0
        done = False

        self.agent = [row, col]

        # Obstacle
        if tuple(self.agent) in self.obstacles:
            reward = -2

        # Pick Item
        elif tuple(self.agent) in self.items and not self.item_collected:
            reward = 2
            self.item_collected = True

        # Goal
        elif tuple(self.agent) == self.goal:
            reward = 5
            done = True

        return np.array(self.agent), reward, done, False, {}

    def render(self):
        grid = np.full((self.size, self.size), ".")

        for obs in self.obstacles:
            grid[obs] = "X"

        for item in self.items:
            if not self.item_collected:
                grid[item] = "I"

        grid[self.goal] = "G"
        grid[self.agent[0], self.agent[1]] = "A"

        print(grid)


# ----------------------------------------
# Policy Evaluation
# ----------------------------------------

def policy_evaluation(env, policy, gamma=0.9, theta=0.001):

    V = np.zeros((env.size, env.size))

    while True:
        delta = 0

        for i in range(env.size):
            for j in range(env.size):

                env.agent = [i, j]
                old_value = V[i][j]

                action = policy[(i, j)]

                next_state, reward, done, _, _ = env.step(action)

                ni, nj = next_state

                if done:
                    new_value = reward
                else:
                    new_value = reward + gamma * V[ni][nj]

                V[i][j] = new_value

                delta = max(delta, abs(old_value - new_value))

        if delta < theta:
            break

    return V


# ----------------------------------------
# Main Program
# ----------------------------------------

env = WarehouseEnv()

# Simple Policy:
# Always Move Right.
# If at last column, Move Down.

policy = {}

for i in range(env.size):
    for j in range(env.size):

        if j < env.size - 1:
            policy[(i, j)] = 3      # Right
        else:
            policy[(i, j)] = 1      # Down


V = policy_evaluation(env, policy)

print("Value Function:\n")
print(np.round(V, 2))

print("\nWarehouse Layout:\n")
env.reset()
env.render()
