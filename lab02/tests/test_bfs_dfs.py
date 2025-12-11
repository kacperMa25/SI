import csv

from boards import *

def experiment():
    sizes = [4, 5, 6, 7, 8]
    results = []

    for size in sizes:
        print(f"\n=== Size {size} ===")

        # BoardPerm BFS
        start = time.perf_counter()
        result, openB, closeB = BoardPerm.BFS(size)
        end = time.perf_counter()
        print(f"BoardPerm BFS: time={end-start:.4f}s, open={openB}, closed={closeB}, solved={result is not None}")
        results.append(("BoardPerm", "BFS", size, end-start, openB, closeB, result is not None))

        # BoardPerm DFS
        start = time.perf_counter()
        result, openD, closeD = BoardPerm.DFS(size)
        end = time.perf_counter()
        print(f"BoardPerm DFS: time={end-start:.4f}s, open={openD}, closed={closeD}, solved={result is not None}")
        results.append(("BoardPerm", "DFS", size, end-start, openD, closeD, result is not None))

        # BoardMatrix BFS
        if size < 7:
            start = time.perf_counter()
            result, openB, closeB = BoardMatrix.BFS(size)
            end = time.perf_counter()
            print(f"BoardMatrix BFS: time={end-start:.4f}s, open={openB}, closed={closeB}, solved={result is not None}")
            results.append(("BoardMatrix", "BFS", size, end-start, openB, closeB, result is not None))

        # BoardMatrix DFS
        start = time.perf_counter()
        result, openD, closeD = BoardMatrix.DFS(size)
        end = time.perf_counter()
        print(f"BoardMatrix DFS: time={end-start:.4f}s, open={openD}, closed={closeD}, solved={result is not None}")
        results.append(("BoardMatrix", "DFS", size, end-start, openD, closeD, result is not None))

    return results

def writeToCSV(results):
    with open("results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["N", "representation", "algorithm", "solution", "open_count", "closed_count", "time"])
        for result in results:
            writer.writerow(result)

if __name__ == "__main__":
    data = experiment()
    writeToCSV(data)
