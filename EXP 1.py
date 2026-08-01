import pandas as pd
import random
import os
from google.colab import files

# ============================
# Upload CSV or Excel File
# ============================
print("Upload your CSV or Excel file")
uploaded = files.upload()

file_name = list(uploaded.keys())[0]

# Read file automatically
if file_name.lower().endswith(".csv"):
    df = pd.read_csv(file_name)
elif file_name.lower().endswith(".xlsx") or file_name.lower().endswith(".xls"):
    df = pd.read_excel(file_name)
else:
    raise Exception("Only CSV and Excel files are supported.")

GRID_SIZE = 5

grid = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

dirt = set()
obstacles = set()
start = (0, 0)

# Build Grid
for _, row in df.iterrows():

    r = int(row["row"])
    c = int(row["col"])
    cell = str(row["type"]).strip().lower()

    if cell == "start":
        start = (r, c)
        grid[r][c] = "S"

    elif cell == "dirt":
        dirt.add((r, c))
        grid[r][c] = "D"

    elif cell == "obstacle":
        obstacles.add((r, c))
        grid[r][c] = "X"

# Actions
actions = [
    (-1,0),
    (1,0),
    (0,-1),
    (0,1)
]

# Check Valid Position
def valid(pos):
    r, c = pos
    return (
        0 <= r < GRID_SIZE and
        0 <= c < GRID_SIZE and
        pos not in obstacles
    )

# Get Neighbours
def neighbours(pos):

    result=[]

    for dr,dc in actions:

        nr=pos[0]+dr
        nc=pos[1]+dc

        if valid((nr,nc)):
            result.append((nr,nc))

    return result

# -----------------------
# Random Policy
# -----------------------
def random_policy():

    position=start
    cleaned=set()
    reward=0
    path=[position]

    for i in range(100):

        if position in dirt and position not in cleaned:
            cleaned.add(position)
            reward+=1

        if len(cleaned)==len(dirt):
            break

        next_cells=neighbours(position)

        if len(next_cells)==0:
            break

        position=random.choice(next_cells)
        path.append(position)

    return reward,path

# -----------------------
# Greedy Policy
# -----------------------
def greedy_policy():

    position=start
    cleaned=set()
    reward=0
    path=[position]

    for i in range(100):

        if position in dirt and position not in cleaned:
            cleaned.add(position)
            reward+=1

        if len(cleaned)==len(dirt):
            break

        next_cells=neighbours(position)

        if len(next_cells)==0:
            break

        best=None
        score=-1

        for cell in next_cells:

            if cell in dirt and cell not in cleaned:
                best=cell
                score=1

        if score==-1:
            best=random.choice(next_cells)

        position=best
        path.append(position)

    return reward,path

# ======================
# Display Grid
# ======================

print("\nGrid Environment\n")

for row in grid:
    print(" ".join(row))

print("\nS = Start")
print("D = Dirt (+1)")
print("X = Obstacle")
print(". = Empty")

# ======================
# Random Policy
# ======================

reward,path=random_policy()

print("\n===== RANDOM POLICY =====")
print("Path:")
print(path)
print("Reward:",reward)

# ======================
# Greedy Policy
# ======================

reward,path=greedy_policy()

print("\n===== GREEDY POLICY =====")
print("Path:")
print(path)
print("Reward:",reward)
