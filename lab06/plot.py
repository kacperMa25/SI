# pyright: standard
import pandas as pd
import matplotlib.pyplot as plt

filenames = [
    "crossover",
    "mutation",
    "generations",
    "population",
    "tournament",
    "baseline_n",
]

for filename in filenames:
    df = pd.read_csv("./csv/" + filename + ".csv")

    df_run_avg = df.groupby(["generation", "parameter"], as_index=False).agg(
        mean_fitness=("mean_fitness", "mean"), best_fitness=("best_fitness", "mean")
    )

    # Mean fitness
    fig, ax = plt.subplots()
    for p in df_run_avg["parameter"].unique():
        subset = df_run_avg[df_run_avg["parameter"] == p]
        ax.plot(subset["generation"], subset["mean_fitness"], label=f"param={p}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean fitness (avg over run)")
    ax.legend()
    fig.savefig("./fig/" + filename + "_mean.svg")
    plt.close(fig)

    # Best fitness
    fig, ax = plt.subplots()
    for p in df_run_avg["parameter"].unique():
        subset = df_run_avg[df_run_avg["parameter"] == p]
        ax.plot(subset["generation"], subset["best_fitness"], label=f"param={p}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (avg over run)")
    ax.legend()
    fig.savefig("./fig/" + filename + "_best.svg")
    plt.close(fig)
