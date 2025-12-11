import pandas as pd
import matplotlib.pyplot as plt

# Load CSV (assuming it's saved as data.csv)
df = pd.read_csv("results.csv")

# Create a line plot of execution_time vs N for each representation_type
plt.figure(figsize=(10, 6))

for rep_type, group in df.groupby("representation_type"):
    plt.plot(group["N"], group["execution_time"], marker="o", label=rep_type)

plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("N (board size)")
plt.ylabel("Execution time [s]")
plt.title("Execution time vs N for different board representations")
plt.legend(title="Representation Type")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
