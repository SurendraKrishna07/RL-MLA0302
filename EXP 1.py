import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class CleaningRobotEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.size = 5

        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)

        # Observation = Robot Position (row, col)
        self.observation_space = spaces.Box(
            low=0,
            high=4,
            shape=(2,),
            dtype=np.int32
        )

        self.initial_dirt = {(0,4), (2,2), (4,1)}
        self.obstacles = {(1,1), (3,3), (4,4)}

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot = [0,0]
        self.dirt = self.initial_dirt.copy()
        return np.array(self.robot), {}

    def step(self, action):

        x, y = self.robot

        if action == 0:      # Up
            x = max(0, x-1)

        elif action == 1:    # Down
            x = min(self.size-1, x+1)

        elif action == 2:    # Left
            y = max(0, y-1)

        elif action == 3:    # Right
            y = min(self.size-1, y+1)

        self.robot = [x,y]

        reward = 0

        if (x,y) in self.obstacles:
            reward = -1

        if (x,y) in self.dirt:
            reward = 1
            self.dirt.remove((x,y))

        done = len(self.dirt) == 0

        return np.array(self.robot), reward, done, False, {}

    def render(self):

        grid = [["." for _ in range(self.size)] for _ in range(self.size)]

        for d in self.dirt:
            grid[d[0]][d[1]] = "D"

        for o in self.obstacles:
            grid[o[0]][o[1]] = "X"

        x,y = self.robot
        grid[x][y] = "R"

        print()

        for row in grid:
            print(" ".join(row))

        print("-"*20)


# --------------------------
# Random Policy
# --------------------------

print("\n===== RANDOM POLICY =====")

env = CleaningRobotEnv()

state, info = env.reset()

total_reward = 0

for i in range(30):

    env.render()

    action = env.action_space.sample()

    state, reward, done, truncated, info = env.step(action)

    total_reward += reward

    print("Action:", action,
          "State:", state,
          "Reward:", reward)

    if done:
        print("All Dirt Cleaned!")
        break

print("Total Reward:", total_reward)

env.close()


# --------------------------
# Greedy Policy
# --------------------------

print("\n===== GREEDY POLICY =====")

env = CleaningRobotEnv()

state, info = env.reset()

total_reward = 0

while True:

    env.render()

    if len(env.dirt) == 0:
        break

    x,y = env.robot

    nearest = min(
        env.dirt,
        key=lambda d: abs(d[0]-x)+abs(d[1]-y)
    )

    dx = nearest[0]-x
    dy = nearest[1]-y

    if abs(dx) > abs(dy):
        action = 1 if dx>0 else 0
    else:
        action = 3 if dy>0 else 2

    state, reward, done, truncated, info = env.step(action)

    total_reward += reward

    print("Action:", action,
          "State:", state,
          "Reward:", reward)

    if done:
        print("All Dirt Cleaned!")
        break

print("Total Reward:", total_reward)

env.close()
