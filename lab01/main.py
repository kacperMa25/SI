import time
import csv
import random

class BoardVector:
    def __init__(self):
        self._size = 4
        self._vector = [random.randint(0, self._size - 1) for _ in range(self._size)]

    def generate_board(self, size):
        self._size = size
        self._vector = [random.randint(0, self._size - 1) for _ in range(self._size)]

    def make_board(self, vector):
        self._size = len(vector)
        self._vector = list(vector)

    def conflicts(self):
        for i in range(self._size):
            for j in range(i + 1, self._size):
                if (
                    abs(self._vector[j] - self._vector[i]) == abs(j - i)
                    or self._vector[i] == self._vector[j]
                ):
                    return True
        return False

    def measure_time(self, iterations):
        cum_time = 0
        conflict_counter = 0
        for _ in range(iterations):
            start = time.perf_counter()
            if self.conflicts():
                conflict_counter += 1
            end = time.perf_counter()
            cum_time += end - start
        conflict_rate = conflict_counter / iterations
        avg = cum_time / iterations
        return avg, conflict_rate


class BoardTuple:
    def __init__(self):
        self._size = 4
        positions = random.sample(range(self._size * self._size), self._size)
        self._tuples = [divmod(p, self._size) for p in positions]

    def generate_board(self, size):
        self._size = size
        positions = random.sample(range(self._size * self._size), self._size)
        self._tuples = [divmod(p, self._size) for p in positions]

    def make_board(self, tuples):
        self._size = len(tuples)
        self._tuples = list(tuples)

    def conflicts(self):
        for i in range(self._size):
            for j in range(i + 1, self._size):
                if (
                    self._tuples[i][0] == self._tuples[j][0]
                    or self._tuples[i][1] == self._tuples[j][1]
                    or abs(self._tuples[i][0] - self._tuples[j][0])
                    == abs(self._tuples[i][1] - self._tuples[j][1])
                ):
                    return True
        return False

    def measure_time(self, iterations):
        cum_time = 0
        conflict_counter = 0
        for _ in range(iterations):
            start = time.perf_counter()
            if self.conflicts():
                conflict_counter += 1
            end = time.perf_counter()
            cum_time += end - start
        conflict_rate = conflict_counter / iterations
        avg = cum_time / iterations
        return avg, conflict_rate


class BoardPerm:
    def __init__(self):
        self._size = 4
        self._perm = list(range(self._size))

    def generate_board(self, size):
        self._size = size
        self._perm = list(range(self._size))
        random.shuffle(self._perm)

    def makeBoard(self, array):
        self._perm = list(array)
        self._size = len(array)

    def printBoard(self):
        matrix = [[0] * self._size for _ in range(self._size)]
        for i in range(self._size):
            matrix[i][self._perm[i]] = 1
        for row in matrix:
            print(row)

    def printPerm(self):
        print(self._perm)

    def conflicts(self):
        for i in range(self._size):
            for j in range(i + 1, self._size):
                if abs(self._perm[j] - self._perm[i]) == abs(j - i):
                    return True
        return False

    def measure_time(self, iterations):
        cum_time = 0
        conflict_counter = 0
        for _ in range(iterations):
            start = time.perf_counter()
            if self.conflicts():
                conflict_counter += 1
            end = time.perf_counter()
            cum_time += end - start
        conflict_rate = conflict_counter / iterations
        avg = cum_time / iterations
        return avg, conflict_rate


class BoardMatrix:
    def __init__(self):
        self._size = 4
        self._matrix = [[0] * self._size for _ in range(self._size)]

    def generate_board(self, size):
        self._size = size
        self._matrix = [[0] * self._size for _ in range(self._size)]
        for col in range(self._size):
            row = random.randint(0, self._size - 1)
            self._matrix[row][col] = 1

    def makeBoard(self, matrix):
        self._matrix = [list(row) for row in matrix]
        self._size = len(matrix)

    def printBoard(self):
        for row in self._matrix:
            print(row)

    def conflicts(self):
        for i in range(self._size):
            row_count = sum(self._matrix[i])
            col_count = sum(self._matrix[j][i] for j in range(self._size))
            if row_count > 1 or col_count > 1:
                return True

        for row in range(self._size):
            for col in range(self._size):
                if self._matrix[row][col] != 1:
                    continue
                for row2 in range(row + 1, self._size):
                    for col2 in range(self._size):
                        if self._matrix[row2][col2] != 1:
                            continue
                        if abs(row - row2) == abs(col - col2):
                            return True
        return False

    def conflictsOptimized(self):
        for i in range(self._size):
            row_count = sum(self._matrix[i])
            col_count = sum(self._matrix[j][i] for j in range(self._size))
            if row_count > 1 or col_count > 1:
                return True

        for row in range(self._size):
            for col in range(self._size):
                if self._matrix[row][col] != 1:
                    continue
                for i in range(1, self._size):
                    if row + i < self._size and col + i < self._size and self._matrix[row + i][col + i] == 1:
                        return True
                    if row + i < self._size and col - i >= 0 and self._matrix[row + i][col - i] == 1:
                        return True
                    if row - i >= 0 and col + i < self._size and self._matrix[row - i][col + i] == 1:
                        return True
                    if row - i >= 0 and col - i >= 0 and self._matrix[row - i][col - i] == 1:
                        return True
        return False

    def measure_time(self, iterations):
        cum_time = 0
        conflict_counter = 0
        for _ in range(iterations):
            start = time.perf_counter()
            if self.conflicts():
                conflict_counter += 1
            end = time.perf_counter()
            cum_time += end - start
        conflict_rate = conflict_counter / iterations
        avg = cum_time / iterations
        return avg, conflict_rate

    def measure_time_optimized(self, iterations):
        cum_time = 0
        conflict_counter = 0
        for _ in range(iterations):
            start = time.perf_counter()
            if self.conflictsOptimized():
                conflict_counter += 1
            end = time.perf_counter()
            cum_time += end - start
        conflict_rate = conflict_counter / iterations
        avg = cum_time / iterations
        return avg, conflict_rate


def experiment(magnitude, iterations):
    boards = [BoardPerm(), BoardMatrix(), BoardTuple(), BoardVector()]

    with open("results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["N", "representation_type", "execution_time", "conflict_rate"])
        for i in range(2, magnitude):
            n = 2 ** i
            for board in boards:
                board.generate_board(n)
                avg_time, conflict_rate = board.measure_time(iterations)
                writer.writerow([n, board.__class__.__name__, avg_time, conflict_rate])
                if board.__class__.__name__ == "BoardMatrix":
                    avg_time, conflict_rate = board.measure_time_optimized(iterations)
                    writer.writerow([n, board.__class__.__name__ + " (optimized)", avg_time, conflict_rate])


def integrityChecks(iterations):
    board_tuple = [(0, 2), (1, 0), (2, 3), (3, 1)]
    boardTuples = BoardTuple()
    boardTuples.make_board(board_tuple)
    checkRep(iterations, boardTuples)

    board_vector = [2, 0, 3, 1]
    boardVectors = BoardVector()
    boardVectors.make_board(board_vector)
    checkRep(iterations, boardVectors)

    board_matrix = [
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
    ]
    boardMatrix = BoardMatrix()
    boardMatrix.makeBoard(board_matrix)
    checkRep(iterations, boardMatrix)

    board_perm = (2, 0, 3, 1)
    boardPerm = BoardPerm()
    boardPerm.makeBoard(board_perm)
    checkRep(iterations, boardPerm)


def checkRep(magnitude, rep):
    start = time.perf_counter()
    print("Poprawne rozwiązanie, czy są konflikty?:", rep.conflicts())
    for i in range(magnitude):
        size = 2 ** i
        rep.generate_board(size)
        print(f"Rozwiązania losowe, czy są konflikty? (dla rozmiaru = {size}): {rep.conflicts()}")
    end = time.perf_counter()
    print(f"Czas reprezentacji {rep.__class__.__name__}: {end - start}")


Magnitude = 16
Iterations = 1000

experiment(Magnitude, Iterations)
integrityChecks(Iterations)
