import csv
import time
from boards import *  # assumes your classes (BoardPerm, BoardMatrix, h1, ...) are here


def run_bestfs_perm(size, hFun):
    start = time.perf_counter()
    result, openC, closeC = BoardPerm.BestFS(size, hFun=hFun)
    end = time.perf_counter()
    return result, openC, closeC, end - start


def run_bestfs_matrix(size, hFun):
    start = time.perf_counter()
    result, closedC, openedC = BoardMatrix.BestFS(size, hFun=hFun)
    end = time.perf_counter()
    return result, openedC, closedC, end - start


def experiment():
    sizes = [4, 5, 6, 7, 8]
    results = []

    heuristics = ["count_attacks", "count_attacks_queens"]

    for size in sizes:
        print(f"\n=== Size {size} ===")

        for h in heuristics:
            result, openBF, closeBF, elapsed = run_bestfs_perm(size, h)
            print(
                f"BoardPerm BestFS ({h}): time={elapsed:.4f}s, open={openBF}, closed={closeBF}, solved={result is not None}"
            )
            results.append(
                (
                    size,
                    "BoardPerm",
                    h,
                    result is not None,
                    openBF,
                    closeBF,
                    elapsed,
                )
            )

            result, openBF_m, closeBF_m, elapsed = run_bestfs_matrix(size, h)
            print(
                f"BoardMatrix BestFS ({h}): time={elapsed:.4f}s, open={openBF_m}, closed={closeBF_m}, solved={result is not None}"
            )
            results.append(
                (
                    size,
                    "BoardMatrix",
                    h,
                    result is not None,
                    openBF_m,
                    closeBF_m,
                    elapsed,
                )
            )
    return results


def writeToCSV(results, filename="results.csv"):
    header = [
        "N",
        "representation",
        "heuristic",
        "solution",
        "open_count",
        "closed_count",
        "time",
    ]

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in results:
            writer.writerow(row)


data = experiment()
writeToCSV(data)
print("\nWyniki zapisane do results_bestfs.csv")
