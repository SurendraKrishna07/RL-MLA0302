import numpy as np

# Grid size
rows, cols = 3, 3

# Goal (Pick-up Point)
goal = (2, 2)

# Discount factor
gamma = 0.9

# Initialize value function
V = np.zeros((rows, cols))

# Possible actions
actions = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

# Reward function
def reward(state):
    if state == goal:
        return 10
    return -1

# Value Iteration
for _ in range(100):
    new_V = V.copy()

    for i in range(rows):
        for j in range(cols):

            if (i, j) == goal:
                continue

            values = []

            for move in actions.values():
                ni = max(0, min(rows - 1, i + move[0]))
                nj = max(0, min(cols - 1, j + move[1]))

                values.append(reward((ni, nj)) + gamma * V[ni][nj])

            new_V[i][j] = max(values)

    if np.max(np.abs(new_V - V)) < 0.001:
        break

    V = new_V

# Find Optimal Policy
policy = np.empty((rows, cols), dtype=object)

for i in range(rows):
    for j in range(cols):

        if (i, j) == goal:
            policy[i][j] = "GOAL"
            continue

        best_action = ""
        best_value = -float("inf")

        for name, move in actions.items():
            ni = max(0, min(rows - 1, i + move[0]))
            nj = max(0, min(cols - 1, j + move[1]))

            value = reward((ni, nj)) + gamma * V[ni][nj]

            if value > best_value:
                best_value = value
                best_action = name

        policy[i][j] = best_action

print("Value Function:")
print(np.round(V, 2))

print("\nOptimal Dispatch Policy:")
for row in policy:
    print(row)
