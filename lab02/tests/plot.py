import pandas as pd
import matplotlib.pyplot as plt

def plot_experiment(csv_file="results.csv"):
    # Load CSV
    df = pd.read_csv(csv_file)

    # Ensure correct column names
    df.columns = ["Representation", "Algorithm", "Size", "Time", "Open", "Closed", "Solved"]

    # Reorder to match logical layout
    df = df[["Representation", "Algorithm", "Size", "Time", "Open", "Closed", "Solved"]]
    df = df.sort_values(["Representation", "Algorithm", "Size"])

    # --- Plot 1: Execution Time ---
    plt.figure(figsize=(10, 6))
    for (rep, alg), subset in df.groupby(["Representation", "Algorithm"]):
        plt.plot(subset["Size"], subset["Time"], marker="o", label=f"{rep} {alg}")
    plt.title("Execution Time vs Board Size")
    plt.xlabel("Board Size (N)")
    plt.ylabel("Execution Time [s]")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Open Count ---
    plt.figure(figsize=(10, 6))
    for (rep, alg), subset in df.groupby(["Representation", "Algorithm"]):
        plt.plot(subset["Size"], subset["Open"], marker="o", label=f"{rep} {alg}")
    plt.title("Open Nodes vs Board Size")
    plt.xlabel("Board Size (N)")
    plt.ylabel("Open Count")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 3: Closed Count ---
    plt.figure(figsize=(10, 6))
    for (rep, alg), subset in df.groupby(["Representation", "Algorithm"]):
        plt.plot(subset["Size"], subset["Closed"], marker="o", label=f"{rep} {alg}")
    plt.title("Closed Nodes vs Board Size")
    plt.xlabel("Board Size (N)")
    plt.ylabel("Closed Count")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_experiment("results.csv")
