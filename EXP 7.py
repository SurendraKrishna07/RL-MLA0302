import numpy as np
import matplotlib.pyplot as plt

# Grid size
rows, cols = 3, 3
goal = (2, 2)

gamma = 0.9

# Initialize value function
V = np.zeros((rows, cols))

# Actions: Up, Down, Left, Right
actions = [(-1,0), (1,0), (0,-1), (0,1)]

# Reward function
def reward(state):
    if state == goal:
        return 10
    return -1

# Bellman Value Iteration
for _ in range(100):
    new_V = V.copy()

    for i in range(rows):
        for j in range(cols):

            if (i, j) == goal:
                continue

            values = []

            for dx, dy in actions:
                ni = max(0, min(rows-1, i + dx))
                nj = max(0, min(cols-1, j + dy))

                values.append(reward((ni, nj)) + gamma * V[ni][nj])

            new_V[i][j] = max(values)

    if np.max(np.abs(new_V - V)) < 0.001:
        break

    V = new_V

print("State Value Function:")
print(np.round(V, 2))

# Visualize Value Function
plt.imshow(V, cmap="viridis")
plt.colorbar(label="State Value")

for i in range(rows):
    for j in range(cols):
        plt.text(j, i, round(V[i][j],1),
                 ha='center', va='center', color='white')

plt.title("Bellman State-Value Function")
plt.show()
