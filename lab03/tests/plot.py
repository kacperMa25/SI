import pandas as pd
import matplotlib.pyplot as plt

# Load CSV

df = pd.read_csv("results.csv", sep=",", engine="python")
print(df.head())
df["N"] = df["N"].astype(int)

plt.figure(figsize=(10, 6))
subset = df[
    (df["representation"] == "BoardPerm")
    & (df["heuristic"].str.contains("count_attacks"))
]

for (rep, heuristic), g in df.groupby(["representation", "heuristic"]):
    plt.plot(g["N"], g["time"], marker="o", label=f"{rep} - {heuristic}")

plt.xlabel("N")
plt.ylabel("Time (s)")
plt.yscale("log")
plt.title("Time vs N (BestFS heuristics, BoardPerm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("1")
plt.show()

# --- Example 2: Nodes opened vs N ---
plt.figure(figsize=(10, 6))
for (rep, heuristic), g in df.groupby(["representation", "heuristic"]):
    plt.plot(g["N"], g["open_count"], marker="o", label=f"{rep} - {heuristic}")

plt.xlabel("N")
plt.ylabel("Open Count")
plt.yscale("log")
plt.title("Open Count vs N")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("2")
plt.show()

# --- Example 3: Closed nodes vs N ---
plt.figure(figsize=(10, 6))
for (rep, heuristic), g in df.groupby(["representation", "heuristic"]):
    plt.plot(g["N"], g["closed_count"], marker="o", label=f"{rep} - {heuristic}")

plt.xlabel("N")
plt.ylabel("Closed Count")
plt.yscale("log")
plt.title("Closed Count vs N")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("3")
plt.show()
